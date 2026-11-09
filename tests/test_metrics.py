from __future__ import annotations

from finnet.metrics import Metrics


def test_metrics_render_and_quantiles() -> None:
    metrics = Metrics()
    metrics.inc("finnet_messages_total")
    metrics.inc("finnet_actions_total", action="ALLOW")
    metrics.gauge("finnet_graph_nodes", 12)
    for value in (1.0, 5.0, 25.0):
        metrics.observe("finnet_e2e_latency_ms", value)
    text = metrics.render_prometheus()
    assert 'finnet_actions_total{action="ALLOW"} 1' in text
    assert "finnet_e2e_latency_ms_bucket" in text
    assert 'le="+Inf"' in text
    assert metrics.quantile("finnet_e2e_latency_ms", 0.5) == 5.0
