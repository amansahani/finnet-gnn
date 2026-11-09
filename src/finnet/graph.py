"""In-memory dynamic multipartite graph with inductive 2-hop neighbourhood export.

Nodes represent IP subnets, device fingerprints, tokenised cards, merchants and
transactions. The graph is deliberately ID-free from the model's perspective:
entity identities are resolved to compact integer IDs per snapshot and the GNN
only sees typed node features and relation-aware message passing. A new entity
can therefore be scored immediately (inductive inference) without retraining.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .config import GraphConfig
from .data.common import TransactionEvent, hour_features
from .features import (
    INPUT_DIM,
    NODE_FEATURE_NAMES,
    NodeType,
    Relation,
    empty_features,
    log1p_safe,
)
from .telemetry import TelemetrySnapshot


@dataclass
class EdgeState:
    relation: Relation
    count: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def weight(self) -> float:
        return min(1.0, 0.35 + 0.2 * math.log1p(self.count))


@dataclass
class TransactionMeta:
    event: TransactionEvent
    telemetry: TelemetrySnapshot
    new_subnet: bool
    new_device: bool
    new_card: bool
    new_merchant: bool
    subnet_card_fanout: int
    device_card_fanout: int
    card_merchant_fanout: int


@dataclass
class NodeState:
    node_id: int
    node_type: NodeType
    key: str
    first_seen: float
    last_seen: float
    tx_count: int = 0
    amount_sum: float = 0.0
    risk_ewma: float = 0.0
    blocked_until: float = 0.0
    counterpart_total: int = 0
    counterparts: dict[NodeType, set[int]] = field(default_factory=dict)
    tx_meta: TransactionMeta | None = None
    deleted: bool = False

    @property
    def avg_amount(self) -> float:
        return self.amount_sum / self.tx_count if self.tx_count else 0.0


@dataclass(frozen=True)
class GraphSnapshot:
    """A detached snapshot safe to hand to a worker thread."""

    x: np.ndarray
    node_types: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_weight: np.ndarray
    target_index: int
    node_ids: np.ndarray
    timestamp: float

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def as_dict(self) -> dict[str, np.ndarray | int | float]:
        return {
            "x": self.x,
            "node_types": self.node_types,
            "edge_index": self.edge_index,
            "edge_type": self.edge_type,
            "edge_weight": self.edge_weight,
            "target_index": self.target_index,
            "node_ids": self.node_ids,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class AddTransactionResult:
    node_id: int
    meta: TransactionMeta


class DynamicTransactionGraph:
    def __init__(self, config: GraphConfig | None = None) -> None:
        self.config = config or GraphConfig()
        self._nodes: dict[int, NodeState] = {}
        self._index: dict[tuple[NodeType, str], int] = {}
        self._adj: dict[int, dict[int, EdgeState]] = {}
        self._tx_queue: deque[int] = deque()
        self._next_id = 0
        self._lock = threading.RLock()
        self._adds = 0

    # ------------------------------------------------------------------ writes
    def add_transaction(
        self,
        event: TransactionEvent,
        telemetry: TelemetrySnapshot,
        *,
        now: float | None = None,
    ) -> AddTransactionResult:
        at = event.timestamp if now is None else now
        with self._lock:
            self._adds += 1
            if self._adds % 256 == 0:
                self._prune_locked(at)

            created: dict[NodeType, bool] = {}
            ids: dict[NodeType, int] = {}
            for node_type, key in (
                (NodeType.SUBNET, event.subnet),
                (NodeType.DEVICE, event.device_id),
                (NodeType.CARD, event.card_token),
                (NodeType.MERCHANT, event.merchant_id),
            ):
                existing = self._index.get((node_type, key))
                created[node_type] = existing is None
                node = self._get_or_create_locked(node_type, key, at)
                ids[node_type] = node.node_id
                node.last_seen = at
                node.tx_count += 1
                node.amount_sum += event.amount

            subnet = self._nodes[ids[NodeType.SUBNET]]
            device = self._nodes[ids[NodeType.DEVICE]]
            card = self._nodes[ids[NodeType.CARD]]
            merchant = self._nodes[ids[NodeType.MERCHANT]]

            # Entity topology, then the transient transaction hub.
            self._link_locked(subnet.node_id, device.node_id, Relation.SUBNET_DEVICE, at)
            self._link_locked(device.node_id, card.node_id, Relation.DEVICE_CARD, at)
            self._link_locked(card.node_id, merchant.node_id, Relation.CARD_MERCHANT, at)

            # Fan-out counters are what make carding visible: one subnet/device
            # touching many cards, or one card touching many merchants.
            self._add_counterpart_locked(subnet, NodeType.DEVICE, device.node_id)
            self._add_counterpart_locked(subnet, NodeType.CARD, card.node_id)
            self._add_counterpart_locked(device, NodeType.SUBNET, subnet.node_id)
            self._add_counterpart_locked(device, NodeType.CARD, card.node_id)
            self._add_counterpart_locked(card, NodeType.SUBNET, subnet.node_id)
            self._add_counterpart_locked(card, NodeType.DEVICE, device.node_id)
            self._add_counterpart_locked(card, NodeType.MERCHANT, merchant.node_id)
            self._add_counterpart_locked(merchant, NodeType.DEVICE, device.node_id)
            self._add_counterpart_locked(merchant, NodeType.CARD, card.node_id)

            tx_node = self._create_locked(
                NodeType.TRANSACTION,
                f"{event.event_id}:{self._next_id}",
                at,
            )
            meta = TransactionMeta(
                event=event,
                telemetry=telemetry,
                new_subnet=created[NodeType.SUBNET],
                new_device=created[NodeType.DEVICE],
                new_card=created[NodeType.CARD],
                new_merchant=created[NodeType.MERCHANT],
                subnet_card_fanout=len(subnet.counterparts.get(NodeType.CARD, ())),
                device_card_fanout=len(device.counterparts.get(NodeType.CARD, ())),
                card_merchant_fanout=len(card.counterparts.get(NodeType.MERCHANT, ())),
            )
            tx_node.tx_meta = meta
            tx_node.tx_count = 1
            tx_node.amount_sum = event.amount
            tx_node.last_seen = at

            for node, relation in (
                (subnet, Relation.TX_SUBNET),
                (device, Relation.TX_DEVICE),
                (card, Relation.TX_CARD),
                (merchant, Relation.TX_MERCHANT),
            ):
                self._link_locked(tx_node.node_id, node.node_id, relation, at)

            self._tx_queue.append(tx_node.node_id)
            self._prune_tx_locked(at)
            return AddTransactionResult(node_id=tx_node.node_id, meta=meta)

    def update_risk(self, tx_node_id: int, score: float, *, now: float | None = None) -> None:
        """Propagate a completed risk score back to entity states.

        The score is deliberately applied after inference so the scoring
        snapshot cannot contain its own label or prediction.
        """

        alpha = self.config.risk_ewma_alpha
        with self._lock:
            tx = self._nodes.get(tx_node_id)
            if tx is None:
                return
            tx.risk_ewma = alpha * score + (1.0 - alpha) * tx.risk_ewma
            for neighbor_id in self._adj.get(tx_node_id, {}):
                node = self._nodes.get(neighbor_id)
                if node is None:
                    continue
                node.risk_ewma = alpha * score + (1.0 - alpha) * node.risk_ewma

    def mark_subnet_blocked(self, subnet: str, until: float) -> bool:
        with self._lock:
            node_id = self._index.get((NodeType.SUBNET, subnet))
            if node_id is None:
                return False
            self._nodes[node_id].blocked_until = until
            return True

    # ------------------------------------------------------------------ read
    def snapshot(
        self,
        target_node_id: int,
        *,
        hops: int | None = None,
        max_nodes: int | None = None,
        now: float | None = None,
    ) -> GraphSnapshot:
        hops = self.config.snapshot_hops if hops is None else hops
        max_nodes = self.config.max_snapshot_nodes if max_nodes is None else max_nodes
        with self._lock:
            target = self._nodes.get(target_node_id)
            if target is None:
                raise KeyError(f"transaction node {target_node_id} no longer exists")
            at = target.last_seen if now is None else now
            selected = self._bfs_locked(target_node_id, hops, max_nodes)
            local = {node_id: i for i, node_id in enumerate(selected)}
            x = np.zeros((len(selected), INPUT_DIM), dtype=np.float32)
            node_types = np.zeros(len(selected), dtype=np.int64)
            for node_id, index in local.items():
                node = self._nodes[node_id]
                x[index] = self._features_locked(node, at)
                node_types[index] = int(node.node_type)

            src: list[int] = []
            dst: list[int] = []
            etype: list[int] = []
            eweight: list[float] = []
            for node_id in selected:
                for neighbor_id, edge in self._adj.get(node_id, {}).items():
                    other = local.get(neighbor_id)
                    if other is None:
                        continue
                    src.append(local[node_id])
                    dst.append(other)
                    etype.append(int(edge.relation))
                    eweight.append(edge.weight)
            edge_index = (
                np.asarray([src, dst], dtype=np.int64)
                if src
                else np.zeros((2, 0), dtype=np.int64)
            )
            return GraphSnapshot(
                x=x,
                node_types=node_types,
                edge_index=edge_index,
                edge_type=np.asarray(etype, dtype=np.int64),
                edge_weight=np.asarray(eweight, dtype=np.float32),
                target_index=local[target_node_id],
                node_ids=np.asarray(selected, dtype=np.int64),
                timestamp=at,
            )

    def node_info(self, node_id: int) -> dict[str, object] | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            return {
                "node_id": node.node_id,
                "type": node.node_type.name,
                "key": node.key,
                "degree": len(self._adj.get(node.node_id, {})),
                "tx_count": node.tx_count,
                "risk_ewma": node.risk_ewma,
                "blocked": node.blocked_until,
            }

    def stats(self) -> dict[str, int]:
        with self._lock:
            by_type = {name: 0 for name in NodeType.__members__}
            for node in self._nodes.values():
                by_type[node.node_type.name] += 1
            edges = sum(len(neighbors) for neighbors in self._adj.values()) // 2
            return {
                "nodes": len(self._nodes),
                "edges": edges,
                "transactions_retained": len(self._tx_queue),
                **{name.lower(): count for name, count in by_type.items()},
            }

    def feature_vector(self, snapshot: GraphSnapshot) -> dict[str, float]:
        values = snapshot.x[snapshot.target_index]
        return {
            name: float(value)
            for name, value in zip(NODE_FEATURE_NAMES, values, strict=True)
        }

    # ------------------------------------------------------------------ internals
    def _get_or_create_locked(self, node_type: NodeType, key: str, at: float) -> NodeState:
        node_id = self._index.get((node_type, key))
        if node_id is not None:
            return self._nodes[node_id]
        return self._create_locked(node_type, key, at)

    def _create_locked(self, node_type: NodeType, key: str, at: float) -> NodeState:
        node_id = self._next_id
        self._next_id += 1
        node = NodeState(
            node_id=node_id,
            node_type=node_type,
            key=key,
            first_seen=at,
            last_seen=at,
        )
        self._nodes[node_id] = node
        self._index[(node_type, key)] = node_id
        self._adj[node_id] = {}
        return node

    def _link_locked(self, a: int, b: int, relation: Relation, at: float) -> None:
        if a == b:
            return
        for src, dst in ((a, b), (b, a)):
            edge = self._adj[src].get(dst)
            if edge is None:
                self._adj[src][dst] = EdgeState(
                    relation=relation, count=1, first_seen=at, last_seen=at
                )
            else:
                edge.count += 1
                edge.last_seen = at

    def _add_counterpart_locked(self, node: NodeState, node_type: NodeType, other_id: int) -> None:
        bucket = node.counterparts.setdefault(node_type, set())
        if other_id in bucket:
            return
        if len(bucket) >= self.config.fanout_cap:
            return
        bucket.add(other_id)
        node.counterpart_total += 1

    def _bfs_locked(self, start: int, hops: int, max_nodes: int) -> list[int]:
        selected = [start]
        seen = {start}
        frontier = [start]
        for _ in range(max(0, hops)):
            next_frontier: list[int] = []
            for node_id in frontier:
                neighbors = sorted(
                    self._adj.get(node_id, {}),
                    key=lambda nid: self._nodes[nid].last_seen,
                    reverse=True,
                )
                for neighbor in neighbors:
                    if neighbor in seen:
                        continue
                    if len(selected) >= max_nodes:
                        return selected
                    seen.add(neighbor)
                    selected.append(neighbor)
                    next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return selected

    def _features_locked(self, node: NodeState, at: float) -> np.ndarray:
        if node.node_type == NodeType.TRANSACTION:
            return self._tx_features_locked(node, at)

        x = empty_features()
        degree = len(self._adj.get(node.node_id, {}))
        x[0] = log1p_safe(degree)
        x[1] = log1p_safe(node.tx_count)
        x[2] = log1p_safe(max(0.0, at - node.last_seen))
        x[3] = log1p_safe(node.avg_amount)
        x[4] = float(np.clip(node.risk_ewma, 0.0, 1.0))
        x[5] = log1p_safe(node.counterpart_total)
        x[6] = 1.0 if at < node.blocked_until else 0.0
        x[7] = log1p_safe(max(0.0, at - node.first_seen))
        return x

    def _tx_features_locked(self, node: NodeState, at: float) -> np.ndarray:
        x = empty_features()
        meta = node.tx_meta
        if meta is None:
            return x
        telemetry = meta.telemetry
        x[0] = log1p_safe(len(self._adj.get(node.node_id, {})))
        x[1] = 1.0
        x[2] = 0.0
        x[3] = log1p_safe(meta.event.amount)
        x[4] = float(np.clip(node.risk_ewma, 0.0, 1.0))
        x[5] = log1p_safe(len(self._adj.get(node.node_id, {})))
        x[6] = 1.0 if at < node.blocked_until else 0.0
        x[7] = 0.0

        hour_sin, hour_cos = hour_features(meta.event.timestamp)
        x[8] = log1p_safe(meta.event.amount)
        x[9] = hour_sin
        x[10] = hour_cos
        x[11] = log1p_safe(telemetry.inter_arrival_ms)
        x[12] = log1p_safe(telemetry.jitter_ms)
        x[13] = log1p_safe(telemetry.burst_velocity)
        x[14] = log1p_safe(telemetry.arrival_rate)
        x[15] = float(np.clip(telemetry.payload_entropy / 8.0, 0.0, 1.0))
        x[16] = log1p_safe(telemetry.rtt_ms)
        x[17] = 1.0 if meta.new_subnet else 0.0
        x[18] = 1.0 if meta.new_device else 0.0
        x[19] = 1.0 if meta.new_card else 0.0
        x[20] = 1.0 if meta.new_merchant else 0.0
        x[21] = log1p_safe(meta.subnet_card_fanout)
        x[22] = log1p_safe(meta.device_card_fanout)
        x[23] = log1p_safe(meta.card_merchant_fanout)
        return x

    def _prune_tx_locked(self, at: float) -> None:
        while self._tx_queue:
            oldest = self._tx_queue[0]
            node = self._nodes.get(oldest)
            if node is None:
                self._tx_queue.popleft()
                continue
            too_old = at - node.last_seen > self.config.tx_ttl_s
            too_many = len(self._tx_queue) > self.config.max_tx_nodes
            if not (too_old or too_many):
                break
            self._tx_queue.popleft()
            self._remove_node_locked(oldest, clean_counterparts=False)

    def _prune_locked(self, at: float) -> None:
        self._prune_tx_locked(at)
        entity_ids = [
            node_id
            for node_id, node in self._nodes.items()
            if node.node_type != NodeType.TRANSACTION
        ]
        excess = len(entity_ids) - self.config.max_entities
        if excess <= 0:
            return
        remove_count = max(excess, max(1, len(entity_ids) // 20))
        victims = sorted(entity_ids, key=lambda nid: self._nodes[nid].last_seen)[:remove_count]
        for node_id in victims:
            self._remove_node_locked(node_id, clean_counterparts=True)

    def _remove_node_locked(self, node_id: int, *, clean_counterparts: bool) -> None:
        node = self._nodes.pop(node_id, None)
        if node is None:
            return
        node.deleted = True
        self._index.pop((node.node_type, node.key), None)
        neighbors = list(self._adj.pop(node_id, {}).keys())
        for neighbor_id in neighbors:
            self._adj.get(neighbor_id, {}).pop(node_id, None)
            if not clean_counterparts:
                continue
            neighbor = self._nodes.get(neighbor_id)
            if neighbor is None:
                continue
            bucket = neighbor.counterparts.get(node.node_type)
            if bucket and node_id in bucket:
                bucket.remove(node_id)
                neighbor.counterpart_total = max(0, neighbor.counterpart_total - 1)

    def replay(
        self,
        events: Iterable[TransactionEvent],
        telemetry_snapshots: Iterable[TelemetrySnapshot],
        *,
        now: float | None = None,
    ) -> None:
        for event, snapshot in zip(events, telemetry_snapshots, strict=True):
            self.add_transaction(event, snapshot, now=now)
