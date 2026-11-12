"""
Automated Multi-Baseline Benchmarking CLI Suite.
Executes empirical evaluation across Baselines 1-3 and Proposed KahnemanGuard architectures.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure workspace root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.clients.tier1_client import Tier1Client
from src.clients.tier2_client import Tier2Client
from src.core.config import settings
from src.core.router import DecisionCategory, DecisionRouter, RoutingAction
from src.engines import create_engine
from src.evaluation.datasets import BenchmarkSample, load_benchmark_suite
from src.evaluation.metrics import SystemMetrics, compute_system_metrics

console = Console()


async def evaluate_fast_only(
    samples: list[BenchmarkSample],
    t1_client: Tier1Client,
    system_name: str,
) -> SystemMetrics:
    """Evaluates single-tier fast model (no confidence escalation)."""
    results = []
    for sample in samples:
        d = await t1_client.evaluate_prompt(sample.prompt)
        action = "BYPASS" if d.category == DecisionCategory.SAFE else "BLOCK"
        p_harm = d.details.get("p_harm", 0.95 if action == "BLOCK" else 0.05)
        results.append({
            "id": sample.id,
            "is_malicious": sample.is_malicious,
            "action": action,
            "latency_ms": d.latency_ms,
            "escalated": False,
            "confidence": d.confidence,
            "prob_harm": p_harm,
        })
    return compute_system_metrics(system_name, results)


async def evaluate_slow_only(
    samples: list[BenchmarkSample],
    t2_client: Tier2Client,
    system_name: str = "Baseline 2 (Slow Only: Llama Guard 3)",
) -> SystemMetrics:
    """Evaluates single-tier slow model (Llama Guard 3 1B on 100% of traffic)."""
    results = []
    for sample in samples:
        d = await t2_client.evaluate_prompt(sample.prompt)
        action = "BYPASS" if d.category == DecisionCategory.SAFE else "BLOCK"
        results.append({
            "id": sample.id,
            "is_malicious": sample.is_malicious,
            "action": action,
            "latency_ms": d.latency_ms,
            "escalated": True,
            "confidence": d.confidence,
            "prob_harm": 0.95 if action == "BLOCK" else 0.05,
        })
    return compute_system_metrics(system_name, results)


async def evaluate_dual_process(
    samples: list[BenchmarkSample],
    t1_client: Tier1Client,
    t2_client: Tier2Client,
    system_name: str,
    tau_safe: float = 0.80,
    tau_unsafe: float = 0.80,
) -> SystemMetrics:
    """Evaluates proposed KahnemanGuard dual-process architecture with confidence routing."""
    router = DecisionRouter(tau_safe=tau_safe, tau_unsafe=tau_unsafe)
    results = []

    for sample in samples:
        t1 = await t1_client.evaluate_prompt(sample.prompt)
        action = router.evaluate(t1)
        p_harm = t1.details.get("p_harm", 0.95 if t1.category != DecisionCategory.SAFE else 0.05)

        if action == RoutingAction.BYPASS:
            results.append({
                "id": sample.id,
                "is_malicious": sample.is_malicious,
                "action": "BYPASS",
                "latency_ms": t1.latency_ms,
                "escalated": False,
                "confidence": t1.confidence,
                "prob_harm": p_harm,
            })
        elif action == RoutingAction.BLOCK:
            results.append({
                "id": sample.id,
                "is_malicious": sample.is_malicious,
                "action": "BLOCK",
                "latency_ms": t1.latency_ms,
                "escalated": False,
                "confidence": t1.confidence,
                "prob_harm": p_harm,
            })
        else:
            # Uncertainty zone -> Escalate to Tier 2 Deep Loop
            t2 = await t2_client.evaluate_prompt(sample.prompt)
            arb = router.arbitrate(t1, t2)
            results.append({
                "id": sample.id,
                "is_malicious": sample.is_malicious,
                "action": arb.action.value,
                "latency_ms": arb.total_latency_ms,
                "escalated": True,
                "confidence": t2.confidence,
                "prob_harm": 0.95 if arb.action == RoutingAction.BLOCK else 0.05,
            })

    return compute_system_metrics(system_name, results)


async def main():
    parser = argparse.ArgumentParser(description="KahnemanGuard Multi-Baseline Benchmarking Suite")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Benchmark scale ('quick': 17 samples, 'full': 1140 samples)")
    parser.add_argument("--variant", choices=["all", "dense", "diffusion", "diffusion-rlcd"], default="all", help="Architecture variant to benchmark ('all', 'dense', 'diffusion', 'diffusion-rlcd')")
    parser.add_argument("--dense-only", action="store_true", help="Shortcut for --variant dense")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per dataset in full mode")
    parser.add_argument("--tau", type=float, default=settings.ROUTER_TAU_SAFE, help=f"Confidence threshold tau (default: {settings.ROUTER_TAU_SAFE})")
    parser.add_argument("--use-onnx", action="store_true", help="Load native ONNX model weights from HUB_DIR specified in .env")
    parser.add_argument("--export", type=str, default="results/benchmark_report.json", help="Path to save JSON benchmark output")
    args = parser.parse_args()

    variant = "dense" if args.dense_only else args.variant
    use_onnx = args.use_onnx or settings.USE_LOCAL_ONNX

    console.print(
        Panel.fit(
            "[bold cyan]KahnemanGuard: A Dual-Process Fast-Slow Architecture for High-Throughput LLM Guardrails[/bold cyan]\n"
            f"[dim]Academic Benchmark Suite | Mode: {args.mode.upper()} | Variant: {variant.upper()} | Threshold Tau: {args.tau} | Use ONNX: {use_onnx}[/dim]",
            border_style="cyan",
        )
    )

    dataset = load_benchmark_suite(mode=args.mode, limit_per_dataset=args.limit)
    n_mal = sum(1 for s in dataset if s.is_malicious)
    n_ben = sum(1 for s in dataset if not s.is_malicious)
    console.print(f"[bold green]Loaded Dataset:[/bold green] {len(dataset)} items ({n_mal} Malicious, {n_ben} Benign)\n")

    t2_deep = Tier2Client(simulate_delay=False)
    all_metrics: list[SystemMetrics] = []

    if variant in ("dense", "all"):
        t1_dense = Tier1Client(engine=create_engine("dense", use_onnx=use_onnx))

    if variant in ("diffusion", "all"):
        t1_diff = Tier1Client(engine=create_engine("diffusion", use_onnx=use_onnx))

    if variant in ("diffusion-rlcd", "all"):
        t1_rlcd = Tier1Client(engine=create_engine("diffusion-rlcd", use_onnx=use_onnx))

    if variant == "dense":
        # 1. Baseline 1: Fast Only (Dense)
        with console.status("[bold yellow]Running Baseline 1 (Fast Only: Dense)...[/bold yellow]"):
            m_b1 = await evaluate_fast_only(dataset, t1_dense, "Baseline 1 (Fast Only: Dense)")
            all_metrics.append(m_b1)

        # 2. Baseline 2: Slow Only (Llama Guard 3 1B)
        with console.status("[bold yellow]Running Baseline 2 (Slow Only: Llama Guard 3)...[/bold yellow]"):
            m_b2 = await evaluate_slow_only(dataset, t2_deep)
            all_metrics.append(m_b2)

        # 3. Proposed: KahnemanGuard (Dense)
        with console.status("[bold yellow]Running Proposed KahnemanGuard (Dense)...[/bold yellow]"):
            m_p_dense = await evaluate_dual_process(dataset, t1_dense, t2_deep, "Proposed KahnemanGuard (Dense)", tau_safe=args.tau, tau_unsafe=args.tau)
            all_metrics.append(m_p_dense)

    elif variant == "diffusion":
        with console.status("[bold yellow]Running Baseline 1 (Fast Only: Diffusion)...[/bold yellow]"):
            m_b1 = await evaluate_fast_only(dataset, t1_diff, "Baseline 1 (Fast Only: Diffusion)")
            all_metrics.append(m_b1)

        with console.status("[bold yellow]Running Baseline 2 (Slow Only: Llama Guard 3)...[/bold yellow]"):
            m_b2 = await evaluate_slow_only(dataset, t2_deep)
            all_metrics.append(m_b2)

        with console.status("[bold yellow]Running Proposed KahnemanGuard (Diffusion)...[/bold yellow]"):
            m_p_diff = await evaluate_dual_process(dataset, t1_diff, t2_deep, "Proposed KahnemanGuard (Diffusion)", tau_safe=args.tau, tau_unsafe=args.tau)
            all_metrics.append(m_p_diff)

    elif variant == "diffusion-rlcd":
        with console.status("[bold yellow]Running Baseline 1 (Fast Only: Diffusion-RLCD)...[/bold yellow]"):
            m_b1 = await evaluate_fast_only(dataset, t1_rlcd, "Baseline 1 (Fast Only: Diffusion-RLCD)")
            all_metrics.append(m_b1)

        with console.status("[bold yellow]Running Baseline 2 (Slow Only: Llama Guard 3)...[/bold yellow]"):
            m_b2 = await evaluate_slow_only(dataset, t2_deep)
            all_metrics.append(m_b2)

        with console.status("[bold yellow]Running Proposed KahnemanGuard (Diffusion-RLCD)...[/bold yellow]"):
            m_p_rlcd = await evaluate_dual_process(dataset, t1_rlcd, t2_deep, "Proposed KahnemanGuard (Diffusion-RLCD)", tau_safe=args.tau, tau_unsafe=args.tau)
            all_metrics.append(m_p_rlcd)

    else:
        # Evaluate All Variants
        with console.status("[bold yellow]Running Baseline 1 (Fast Only: ModernBERT Dense)...[/bold yellow]"):
            m_b1 = await evaluate_fast_only(dataset, t1_dense, "Baseline 1 (Fast: ModernBERT Dense)")
            all_metrics.append(m_b1)

        with console.status("[bold yellow]Running Baseline 2 (Fast Only: ModernBERT Diffusion)...[/bold yellow]"):
            m_b2 = await evaluate_fast_only(dataset, t1_diff, "Baseline 2 (Fast: ModernBERT Diffusion)")
            all_metrics.append(m_b2)

        with console.status("[bold yellow]Running Baseline 3 (Slow Only: Llama Guard 3 1B)...[/bold yellow]"):
            m_b3 = await evaluate_slow_only(dataset, t2_deep, system_name="Baseline 3 (Slow Only: Llama Guard 3 1B)")
            all_metrics.append(m_b3)

        with console.status("[bold yellow]Running Proposed KahnemanGuard (Dense + Router)...[/bold yellow]"):
            m_p_dense = await evaluate_dual_process(dataset, t1_dense, t2_deep, "Proposed KahnemanGuard (Dense)", tau_safe=args.tau, tau_unsafe=args.tau)
            all_metrics.append(m_p_dense)

        with console.status("[bold yellow]Running Proposed KahnemanGuard (Diffusion + Router)...[/bold yellow]"):
            m_p_diff = await evaluate_dual_process(dataset, t1_diff, t2_deep, "Proposed KahnemanGuard (Diffusion)", tau_safe=args.tau, tau_unsafe=args.tau)
            all_metrics.append(m_p_diff)

        with console.status("[bold yellow]Running Proposed KahnemanGuard (Diffusion-RLCD + Router)...[/bold yellow]"):
            m_p_rlcd = await evaluate_dual_process(dataset, t1_rlcd, t2_deep, "Proposed KahnemanGuard (Diffusion-RLCD)", tau_safe=args.tau, tau_unsafe=args.tau)
            all_metrics.append(m_p_rlcd)

    # Render Summary Table
    table = Table(
        title=f"KahnemanGuard Empirical Benchmark Matrix (N={len(dataset)}, Tau={args.tau})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Architecture / System", style="cyan", width=38)
    table.add_column("DSR", justify="right", style="green")
    table.add_column("FPR", justify="right", style="yellow")
    table.add_column("Brier", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right", style="bold red")
    table.add_column("Escalation", justify="right")
    table.add_column("Savings", justify="right", style="bold green")

    for m in all_metrics:
        table.add_row(
            m.system_name,
            f"{m.defense_success_rate_pct:.1f}%",
            f"{m.false_positive_rate_pct:.1f}%",
            f"{m.brier_score:.4f}",
            f"{m.expected_calibration_error:.4f}",
            f"{m.p50_latency_ms:.1f}ms",
            f"{m.p99_latency_ms:.1f}ms",
            f"{m.escalation_rate_pct:.1f}%",
            f"{m.cost_savings_pct:.1f}%",
        )

    console.print(table)

    # Export JSON results
    export_path = Path(args.export)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "benchmark_mode": args.mode,
        "sample_count": len(dataset),
        "tau": args.tau,
        "systems": [m.model_dump() for m in all_metrics]
    }
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    console.print(f"\n[bold green][OK] Benchmark JSON report successfully exported to:[/bold green] {export_path}")

    # Export Markdown results
    md_path = export_path.with_suffix(".md")
    md_lines = [
        f"# KahnemanGuard Empirical Benchmark Report",
        f"- **Benchmark Mode**: `{args.mode.upper()}`",
        f"- **Sample Count**: `{len(dataset)}`",
        f"- **Confidence Threshold (tau)**: `{args.tau}`\n",
        "| Architecture / System | DSR (↑) | FPR (↓) | Brier (↓) | ECE (↓) | P50 (ms) | P99 (ms) | Escalation | Savings (↑) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in all_metrics:
        md_lines.append(
            f"| **{m.system_name}** | {m.defense_success_rate_pct:.1f}% | {m.false_positive_rate_pct:.1f}% | "
            f"{m.brier_score:.4f} | {m.expected_calibration_error:.4f} | {m.p50_latency_ms:.1f}ms | "
            f"{m.p99_latency_ms:.1f}ms | {m.escalation_rate_pct:.1f}% | {m.cost_savings_pct:.1f}% |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    console.print(f"[bold green][OK] Benchmark Markdown report successfully exported to:[/bold green] {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
