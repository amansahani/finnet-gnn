"""Command line interface for training, serving, simulation and benchmarking."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .attacks import evaluate_adversarial
from .baselines import run_baseline_benchmark
from .calibration import Calibrator
from .config import AppConfig, load_config
from .data.synthetic import event_to_row, generate_synthetic_events
from .evaluation import evaluate_scores
from .features import NODE_FEATURE_NAMES
from .graph import DynamicTransactionGraph
from .model import load_model_checkpoint
from .scorer import RiskScorer
from .server import GatewayServer
from .simulator import SimulatorConfig, run_simulation
from .telemetry import FlowTelemetry
from .training import evaluate_model, load_events_for_training, train_from_events


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should show a concise failure
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finnet",
        description="FinNet-GNN autonomous ISO-8583 fraud gateway",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML configuration path (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train the dynamic GNN")
    _common_config(train)
    train.add_argument("--source", choices=["synthetic", "ieee"], default=None)
    train.add_argument("--events", type=int, default=None, help="Synthetic event count")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--max-events", type=int, default=None)
    train.add_argument("--transactions", default=None, help="IEEE train_transaction.csv")
    train.add_argument("--identity", default=None, help="IEEE train_identity.csv")
    train.add_argument("--output", default=None, help="Checkpoint path")
    train.add_argument("--risk-gate", action="store_true", help="Enable risk-gated message passing")
    train.add_argument("--temporal-encoding", action="store_true")
    train.add_argument("--time-dim", type=int, default=None)
    train.add_argument("--temporal-memory", action="store_true")
    train.add_argument("--simulate-rejection", action="store_true")
    train.add_argument("--json-out", default=None)
    train.set_defaults(func=_cmd_train)

    serve = sub.add_parser("serve", help="Run the gateway")
    _common_config(serve)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--metrics-port", type=int, default=None)
    serve.add_argument("--model", default=None)
    serve.add_argument("--log-path", default=None)
    serve.add_argument("--no-uvloop", action="store_true")
    serve.add_argument(
        "--trust-client-subnet",
        action="store_true",
        help="TEST ONLY: trust ISO field 61 claimed source subnet (for the simulator)",
    )
    serve.add_argument("--enable-os-firewall", action="store_true")
    serve.add_argument("--firewall-backend", choices=["audit", "pf", "iptables"], default=None)
    serve.add_argument("--calibrator", default=None, help="Path to JSON calibrator written by `finnet calibrate`")
    serve.add_argument("--shadow-mode", action="store_true", help="Compute actions but force ALLOW responses")
    serve.set_defaults(func=_cmd_serve)

    simulate = sub.add_parser("simulate", help="Send legitimate and attack traffic")
    simulate.add_argument("--host", default="127.0.0.1")
    simulate.add_argument("--port", type=int, default=8583)
    simulate.add_argument("--events", type=int, default=600)
    simulate.add_argument("--fraud-share", type=float, default=0.55)
    simulate.add_argument("--seed", type=int, default=11)
    simulate.add_argument("--json-out", default=None)
    simulate.set_defaults(func=_cmd_simulate)

    demo = sub.add_parser("demo", help="Train if needed, start gateway, and simulate attacks")
    _common_config(demo)
    demo.add_argument("--events", type=int, default=450, help="Simulated transactions")
    demo.add_argument("--train-events", type=int, default=12000)
    demo.add_argument("--epochs", type=int, default=10)
    demo.add_argument("--retrain", action="store_true")
    demo.add_argument("--no-train", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    gen = sub.add_parser("gen-data", help="Generate a synthetic CSV stream")
    gen.add_argument("--events", type=int, default=12000)
    gen.add_argument("--fraud-rate", type=float, default=0.08)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--out", default="data/synthetic_transactions.csv")
    gen.set_defaults(func=_cmd_gen_data)

    bench = sub.add_parser("benchmark", help="Measure graph-export + inference latency")
    _common_config(bench)
    bench.add_argument("--model", default=None)
    bench.add_argument("--events", type=int, default=5000)
    bench.add_argument("--iterations", type=int, default=1000)
    bench.add_argument("--json-out", default=None)
    bench.set_defaults(func=_cmd_benchmark)

    download = sub.add_parser("download-data", help="Download IEEE-CIS via Kaggle CLI")
    download.add_argument("--dest", default="data")
    download.set_defaults(func=_cmd_download)

    baseline = sub.add_parser(
        "baseline-run", help="Run classical baselines and compare with the GNN"
    )
    _common_config(baseline)
    baseline.add_argument("--source", choices=["synthetic", "ieee"], default=None)
    baseline.add_argument("--events", type=int, default=None, help="Synthetic event count")
    baseline.add_argument("--max-events", type=int, default=None)
    baseline.add_argument("--transactions", default=None)
    baseline.add_argument("--identity", default=None)
    baseline.add_argument("--output", default=None, help="Reserved for future output model path")
    baseline.add_argument(
        "--protocol",
        choices=["temporal", "card-holdout", "device-holdout", "merchant-holdout"],
        default="temporal",
    )
    baseline.add_argument("--model", default=None)
    baseline.add_argument("--train-epochs", type=int, default=None)
    baseline.add_argument(
        "--calibration-method",
        choices=["none", "isotonic", "platt", "temperature"],
        default="isotonic",
    )
    baseline.add_argument("--false-positive-cost", type=float, default=1.0)
    baseline.add_argument("--false-negative-cost", type=float, default=10.0)
    baseline.add_argument("--json-out", default="artifacts/baseline_report.json")
    baseline.set_defaults(func=_cmd_baseline)

    attack = sub.add_parser(
        "attack-eval", help="Evaluate robustness under constrained evasive attacks"
    )
    _common_config(attack)
    attack.add_argument("--source", choices=["synthetic", "ieee"], default=None)
    attack.add_argument("--events", type=int, default=None, help="Synthetic event count")
    attack.add_argument("--max-events", type=int, default=None)
    attack.add_argument("--transactions", default=None)
    attack.add_argument("--identity", default=None)
    attack.add_argument("--model", default=None)
    attack.add_argument(
        "--attack",
        action="append",
        choices=[
            "all",
            "amount_noise",
            "merchant_rotation",
            "device_rotation",
            "subnet_hopping",
            "amount_split",
            "time_scatter",
        ],
        default=None,
    )
    attack.add_argument("--json-out", default="artifacts/attack_report.json")
    attack.set_defaults(func=_cmd_attack)

    calibrate = sub.add_parser("calibrate", help="Fit a JSON calibrator from a GNN checkpoint")
    _common_config(calibrate)
    calibrate.add_argument("--source", choices=["synthetic", "ieee"], default=None)
    calibrate.add_argument("--events", type=int, default=None, help="Synthetic event count")
    calibrate.add_argument("--max-events", type=int, default=None)
    calibrate.add_argument("--transactions", default=None)
    calibrate.add_argument("--identity", default=None)
    calibrate.add_argument("--model", default=None)
    calibrate.add_argument(
        "--method", choices=["isotonic", "platt", "temperature"], default="isotonic"
    )
    calibrate.add_argument("--output", default="artifacts/calibrator.json")
    calibrate.add_argument("--json-out", default=None)
    calibrate.set_defaults(func=_cmd_calibrate)

    info = sub.add_parser("info", help="Show effective configuration and model status")
    _common_config(info)
    info.add_argument("--model", default=None)
    info.set_defaults(func=_cmd_info)
    return parser


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=argparse.SUPPRESS, help="YAML configuration path")


def _config_from_args(args: argparse.Namespace, **sections: dict[str, Any]) -> AppConfig:
    path = getattr(args, "config", "configs/default.yaml")
    config_path = path if Path(path).exists() else None
    return load_config(config_path, overrides={k: v for k, v in sections.items() if v is not None})


def _rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except Exception:  # noqa: BLE001 - best-effort memory introspection
        return 0.0


def _cmd_train(args: argparse.Namespace) -> int:
    sections: dict[str, Any] = {}
    if args.source:
        sections["data"] = {"source": args.source}
    if args.events:
        sections.setdefault("data", {})["synthetic_events"] = args.events
    if args.max_events:
        sections.setdefault("data", {})["max_events"] = args.max_events
    if args.transactions:
        sections.setdefault("data", {})["transactions_csv"] = args.transactions
    if args.identity:
        sections.setdefault("data", {})["identity_csv"] = args.identity
    if args.output:
        sections.setdefault("model", {})["checkpoint"] = args.output
    if args.risk_gate:
        sections.setdefault("model", {})["risk_gate"] = True
    if args.temporal_encoding:
        sections.setdefault("model", {})["temporal_encoding"] = True
    if args.time_dim is not None:
        sections.setdefault("model", {})["time_dim"] = args.time_dim
    if args.temporal_memory:
        sections.setdefault("model", {})["temporal_memory"] = True
    if args.simulate_rejection:
        sections.setdefault("data", {})["simulate_rejection"] = True
    config = _config_from_args(args, **sections)
    events, source = load_events_for_training(config)
    print(f"loaded {len(events):,} events from {source}", flush=True)
    metrics = train_from_events(
        events,
        config,
        output_path=args.output or config.model.checkpoint,
        source_name=source,
        epochs=args.epochs,
        progress=lambda message: print(message, flush=True),
    )
    summary = {
        key: metrics[key]
        for key in ("auc_roc", "auc_pr", "accuracy_at_0.5", "precision_at_0.5", "recall_at_0.5")
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    sections: dict[str, Any] = {}
    if args.host:
        sections.setdefault("server", {})["host"] = args.host
    if args.port is not None:
        sections.setdefault("server", {})["port"] = args.port
    if args.metrics_port is not None:
        sections.setdefault("server", {})["metrics_port"] = args.metrics_port
    if args.model:
        sections.setdefault("model", {})["checkpoint"] = args.model
    if args.log_path:
        sections["log_path"] = args.log_path
    if args.enable_os_firewall:
        sections.setdefault("policy", {})["enable_os_firewall"] = True
    if args.firewall_backend:
        sections.setdefault("policy", {})["firewall_backend"] = args.firewall_backend
    if args.calibrator:
        sections.setdefault("policy", {})["calibration_path"] = args.calibrator
    if args.shadow_mode:
        sections.setdefault("policy", {})["shadow_mode"] = True
    if args.no_uvloop:
        sections.setdefault("server", {})["use_uvloop"] = False
    if args.trust_client_subnet:
        sections.setdefault("server", {})["trust_client_subnet"] = True
    config = _config_from_args(args, **sections)
    return _run_server(config)


def _run_server(config: AppConfig) -> int:
    if config.server.use_uvloop:
        try:
            import uvloop

            uvloop.install()
        except ImportError:
            pass

    async def runner() -> None:
        server = GatewayServer(config)
        await server.start()
        engine = server.engine
        print(
            f"FinNet-GNN listening on tcp://{config.server.host}:{server.port} "
            f"metrics=http://{config.server.metrics_host}:{server.metrics_port}",
            flush=True,
        )
        if not engine.scorer.model_loaded:
            print("warning: no GNN checkpoint; using heuristic fallback scorer", file=sys.stderr)
            print(f"         ({engine.scorer.load_error})", file=sys.stderr)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, server.request_stop)
            except NotImplementedError:
                pass
        await server.serve_forever()
        await server.stop()
        print("gateway stopped", flush=True)

    asyncio.run(runner())
    return 0


def _baseline_config_from_args(args: argparse.Namespace):
    sections: dict[str, Any] = {}
    if getattr(args, "source", None):
        sections.setdefault("data", {})["source"] = args.source
    if getattr(args, "events", None):
        sections.setdefault("data", {})["synthetic_events"] = args.events
    if getattr(args, "max_events", None):
        sections.setdefault("data", {})["max_events"] = args.max_events
    if getattr(args, "transactions", None):
        sections.setdefault("data", {})["transactions_csv"] = args.transactions
    if getattr(args, "identity", None):
        sections.setdefault("data", {})["identity_csv"] = args.identity
    if getattr(args, "model", None):
        sections.setdefault("model", {})["checkpoint"] = args.model
    return _config_from_args(args, **sections)


def _cmd_baseline(args: argparse.Namespace) -> int:
    config = _baseline_config_from_args(args)
    events, source = load_events_for_training(config)
    print(f"loaded {len(events):,} events from {source}", flush=True)
    report = run_baseline_benchmark(
        events,
        config,
        source_name=source,
        model_path=args.model or config.model.checkpoint,
        train_epochs=args.train_epochs,
        calibration_method=args.calibration_method,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
        protocol=args.protocol,
    )
    compact = _baseline_summary(report)
    print(json.dumps(compact, indent=2))
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _baseline_summary(report: dict[str, Any]) -> dict[str, Any]:
    baselines = report.get("baselines", {})
    summary: dict[str, Any] = {}
    for name, payload in baselines.items():
        if not isinstance(payload, dict) or not payload.get("available", False):
            continue
        raw = payload.get("raw", {})
        summary[name] = {
            key: raw.get(key)
            for key in (
                "auc_roc",
                "average_precision",
                "precision_at_0.5",
                "recall_at_0.5",
                "expected_cost_at_0.5",
            )
        }
        if "calibrated" in payload:
            calibrated = payload["calibrated"]
            summary[name]["calibrated_average_precision"] = calibrated.get("average_precision")
            summary[name]["calibrated_ece"] = calibrated.get("calibration", {}).get("ece")
    return summary


def _cmd_calibrate(args: argparse.Namespace) -> int:
    config = _baseline_config_from_args(args)
    events, source = load_events_for_training(config)
    checkpoint = Path(args.model or config.model.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    from .training import make_datasets

    train_samples, test_samples, _, _ = make_datasets(events, config)
    model, _ = load_model_checkpoint(str(checkpoint), config.model.device)
    train_probabilities, train_labels = evaluate_model(
        model, train_samples, batch_size=256, device=config.model.device
    )
    test_probabilities, test_labels = evaluate_model(
        model, test_samples, batch_size=256, device=config.model.device
    )
    calibrator = Calibrator.fit(train_probabilities, train_labels, args.method)
    calibrator.save(args.output)
    calibrated_test = calibrator.predict(test_probabilities)
    report = {
        "source": source,
        "model": str(checkpoint),
        "method": args.method,
        "output": str(args.output),
        "before": evaluate_scores(test_labels, test_probabilities),
        "after": evaluate_scores(test_labels, calibrated_test),
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _cmd_attack(args: argparse.Namespace) -> int:
    config = _baseline_config_from_args(args)
    events, source = load_events_for_training(config)
    checkpoint = Path(args.model or config.model.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    selected = args.attack or ["all"]
    transformations = None if "all" in selected else selected
    print(f"loaded {len(events):,} events from {source}", flush=True)
    report = evaluate_adversarial(
        events,
        config,
        model_path=str(checkpoint),
        transformations=transformations,
        max_events=config.data.attack_eval_max_events,
    )
    report["source"] = source
    print(json.dumps(report, indent=2))
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    report = asyncio.run(
        run_simulation(
            SimulatorConfig(
                host=args.host,
                port=args.port,
                events=args.events,
                fraud_share=args.fraud_share,
                seed=args.seed,
            )
        )
    )
    _print_report(report.as_dict())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    checkpoint = Path(config.model.checkpoint)
    if args.retrain or (not checkpoint.exists() and not args.no_train):
        print(
            f"no checkpoint found; training a demo model on {args.train_events:,} synthetic events",
            flush=True,
        )
        train_config = replace(
            config,
            data=replace(config.data, source="synthetic", synthetic_events=args.train_events),
            model=replace(config.model, epochs=args.epochs, checkpoint=config.model.checkpoint),
        )
        events = generate_synthetic_events(
            args.train_events,
            fraud_rate=train_config.data.synthetic_fraud_rate,
            seed=train_config.data.synthetic_seed,
        )
        train_from_events(
            events,
            train_config,
            output_path=train_config.model.checkpoint,
            source_name="synthetic",
            epochs=args.epochs,
            progress=lambda message: print(message, flush=True),
        )
    config = replace(
        config,
        server=replace(
            config.server,
            port=0,
            metrics_port=0,
            use_uvloop=False,
            trust_client_subnet=True,
        ),
    )

    async def runner() -> None:
        server = GatewayServer(config)
        await server.start()
        try:
            print(
                f"demo gateway listening on tcp://127.0.0.1:{server.port} "
                f"metrics=http://127.0.0.1:{server.metrics_port}",
                flush=True,
            )
            report = await run_simulation(
                SimulatorConfig(port=server.port, events=args.events, fraud_share=0.6)
            )
            _print_report(report.as_dict())
            await asyncio.sleep(0.2)
            status = server.engine.status()
            print(json.dumps(status, indent=2, default=str))
            print(
                f"model={status['model_source']} graph={status['graph']['nodes']} nodes/"
                f"{status['graph']['edges']} edges "
                f"quarantined={status['policy']['quarantined_subnets']}"
            )
        finally:
            await server.stop()

    asyncio.run(runner())
    return 0


def _cmd_gen_data(args: argparse.Namespace) -> int:
    events = generate_synthetic_events(
        args.events, fraud_rate=args.fraud_rate, seed=args.seed
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [event_to_row(event) for event in events]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):,} rows to {output}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    sections: dict[str, Any] = {}
    if args.model:
        sections.setdefault("model", {})["checkpoint"] = args.model
    config = _config_from_args(args, **sections)
    events = generate_synthetic_events(
        max(args.events + 1000, 2000),
        fraud_rate=config.data.synthetic_fraud_rate,
        seed=config.data.synthetic_seed,
    )
    graph = DynamicTransactionGraph(config.graph)
    telemetry = FlowTelemetry(config.telemetry)
    scorer = RiskScorer(config.model, config.model.checkpoint)
    for event in events[:1000]:
        snapshot = telemetry.update(
            event.flow_key,
            event.raw_payload,
            at=event.timestamp,
            rtt_ms=event.rtt_ms,
        )
        added = graph.add_transaction(event, snapshot, now=event.timestamp)
        graph.update_risk(added.node_id, 0.1, now=event.timestamp)

    end_to_end: list[float] = []
    inference: list[float] = []
    over_budget = 0
    for event in events[1000 : 1000 + args.iterations]:
        started = time.perf_counter()
        snapshot = telemetry.update(
            event.flow_key,
            event.raw_payload,
            at=event.timestamp,
            rtt_ms=event.rtt_ms,
        )
        added = graph.add_transaction(event, snapshot, now=event.timestamp)
        graph_snapshot = graph.snapshot(added.node_id, now=event.timestamp)
        score = scorer.predict(graph_snapshot)
        elapsed = (time.perf_counter() - started) * 1000.0
        end_to_end.append(elapsed)
        inference.append(score.latency_ms)
        if not score.within_budget:
            over_budget += 1
    end_to_end.sort()
    inference.sort()
    report = {
        "model": "gnn" if scorer.model_loaded else "heuristic",
        "iterations": len(end_to_end),
        "feature_dim": len(NODE_FEATURE_NAMES),
        "p50_e2e_ms": _quantile(end_to_end, 0.50),
        "p95_e2e_ms": _quantile(end_to_end, 0.95),
        "p99_e2e_ms": _quantile(end_to_end, 0.99),
        "p50_inference_ms": _quantile(inference, 0.50),
        "p95_inference_ms": _quantile(inference, 0.95),
        "p99_inference_ms": _quantile(inference, 0.99),
        "within_20ms_pct": 100.0
        * sum(1 for value in end_to_end if value <= 20.0)
        / max(1, len(end_to_end)),
        "budget_fallbacks": over_budget,
        "rss_mb": _rss_mb(),
        "graph": graph.stats(),
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .data.ieee_cis import download_ieee_cis

    destination = download_ieee_cis(args.dest)
    print(f"IEEE-CIS files extracted under {destination}")
    print("Set data.source=ieee and run: finnet train --source ieee")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    sections: dict[str, Any] = {}
    if args.model:
        sections.setdefault("model", {})["checkpoint"] = args.model
    config = _config_from_args(args, **sections)
    scorer = RiskScorer(config.model, config.model.checkpoint)
    payload = {
        "config": config.to_dict(),
        "checkpoint": config.model.checkpoint,
        "checkpoint_exists": Path(config.model.checkpoint).exists(),
        "model_loaded": scorer.model_loaded,
        "model_load_error": scorer.load_error,
        "feature_names": list(NODE_FEATURE_NAMES),
    }
    if scorer.model_loaded:
        payload["model_parameters"] = scorer.model.parameter_count() if scorer.model else 0
        payload["checkpoint_metrics"] = scorer.checkpoint_meta.get("metrics", {})
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(len(sorted_values) * q) - 1))
    return round(sorted_values[index], 3)


if __name__ == "__main__":
    raise SystemExit(main())
