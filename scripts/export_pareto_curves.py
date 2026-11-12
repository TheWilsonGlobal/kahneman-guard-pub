"""
Pareto Optimization & Threshold Sensitivity Sweep.
Sweeps confidence routing threshold tau across [0.50, 0.95] to construct
Pareto frontiers (DSR vs Escalation Rate, Latency vs Cost Savings).
"""

import asyncio
import csv
import json
import sys
from pathlib import Path

# Ensure workspace root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.clients.tier1_client import Tier1Client
from src.clients.tier2_client import Tier2Client
from src.engines import create_engine
from src.evaluation.datasets import load_benchmark_suite
from scripts.run_benchmark import evaluate_dual_process


async def sweep_pareto(mode: str = "quick", limit: int | None = 50):
    print("================================================================")
    print(" KAHNEMANGUARD: CONFIDENCE THRESHOLD PARETO FRONTIER SWEEPER    ")
    print("================================================================")

    dataset = load_benchmark_suite(mode=mode, limit_per_dataset=limit)
    print(f"Loaded {len(dataset)} evaluation samples for Pareto sweep.\n")

    t1_dense = Tier1Client(engine=create_engine("dense"))
    t1_diff = Tier1Client(engine=create_engine("diffusion"))
    t1_rlcd = Tier1Client(engine=create_engine("diffusion-rlcd"))
    t2_deep = Tier2Client(simulate_delay=False)

    tau_values = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    models = [
        ("KahnemanGuard-Dense", t1_dense),
        ("KahnemanGuard-Diffusion", t1_diff),
        ("KahnemanGuard-Diffusion-RLCD", t1_rlcd),
    ]

    records = []

    for tau in tau_values:
        print(f"--> Sweeping threshold tau = {tau:.2f}...")
        for name, t1 in models:
            metrics = await evaluate_dual_process(
                dataset, t1, t2_deep, system_name=name, tau_safe=tau, tau_unsafe=tau
            )
            records.append({
                "model": name,
                "tau": tau,
                "dsr_pct": metrics.defense_success_rate_pct,
                "fpr_pct": metrics.false_positive_rate_pct,
                "brier_score": metrics.brier_score,
                "ece": metrics.expected_calibration_error,
                "p50_latency_ms": metrics.p50_latency_ms,
                "p99_latency_ms": metrics.p99_latency_ms,
                "escalation_rate_pct": metrics.escalation_rate_pct,
                "cost_savings_pct": metrics.cost_savings_pct,
            })

    # Save to CSV
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "pareto_curves.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Saved Pareto sweep metrics to: {csv_path}")

    # Save to JSON
    json_path = results_dir / "pareto_curves.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[OK] Saved JSON curves to: {json_path}")

    # Generate figures with matplotlib
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        colors = {
            "KahnemanGuard-Dense": "#1f77b4",
            "KahnemanGuard-Diffusion": "#ff7f0e",
            "KahnemanGuard-Diffusion-RLCD": "#2ca02c",
        }

        for name, _ in models:
            sub = [r for r in records if r["model"] == name]
            taus = [r["tau"] for r in sub]
            dsrs = [r["dsr_pct"] for r in sub]
            escs = [r["escalation_rate_pct"] for r in sub]
            lats = [r["p50_latency_ms"] for r in sub]
            savs = [r["cost_savings_pct"] for r in sub]

            ax1.plot(escs, dsrs, marker="o", label=name, color=colors.get(name, "black"))
            ax2.plot(taus, savs, marker="s", label=name, color=colors.get(name, "black"))

        ax1.set_title("Pareto Frontier: Security (DSR) vs Escalation Rate", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Tier 2 Escalation Rate (%)", fontsize=10)
        ax1.set_ylabel("Defense Success Rate (DSR %)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(fontsize=9)

        ax2.set_title("Compute Cost Savings vs Threshold Tau", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Routing Threshold Tau", fontsize=10)
        ax2.set_ylabel("Compute Cost Savings (%)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(fontsize=9)

        fig.tight_layout()
        plot_path = results_dir / "pareto_frontiers.png"
        fig.savefig(plot_path, dpi=200)
        print(f"[OK] Rendered Pareto publication figure: {plot_path}")
    except Exception as e:
        print(f"Notice: Matplotlib plot export encountered: {e}")


def main():
    asyncio.run(sweep_pareto(mode="quick"))


if __name__ == "__main__":
    main()
