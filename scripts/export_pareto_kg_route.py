"""
Pareto Optimization & Dual-Threshold Routing Sweeper on Frozen Benchmark KG-Eval Route (N=400).
Evaluates trade-off curves between Escalation Rate (Compute Cost) and Security (DSR / FPR).
Exports results to JSON, CSV, and renders publication-ready Pareto frontiers PNG.
"""

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from src.clients.tier2_client import Tier2Client
from src.core.router import DecisionCategory, DecisionRouter, RoutingAction, RoutingResult
from src.engines import create_engine


async def sweep_pareto_kg_route(
    benchmark_path: str = "data/benchmarks/kg_eval_route.jsonl",
    tier2_cache_path: str = "results/tier2_route_eval.json",
    output_json_path: str = "results/pareto_curves_route.json",
    output_csv_path: str = "results/pareto_curves_route.csv",
    output_plot_path: str = "results/pareto_frontiers_route.png",
):
    print("=" * 70)
    print(" PARETO THRESHOLD OPTIMIZATION ON KG-EVAL ROUTE (N=400)")
    print("=" * 70)

    bench_file = ROOT_DIR / benchmark_path
    if not bench_file.exists():
        raise FileNotFoundError(f"Benchmark not found: {bench_file}")

    samples = []
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    n_samples = len(samples)
    total_malicious = sum(1 for s in samples if s["is_malicious"])
    total_benign = sum(1 for s in samples if not s["is_malicious"])
    print(f"--> Loaded {n_samples} samples (Malicious: {total_malicious}, Benign: {total_benign})")

    # 1. Evaluate System 1 (ModernBERT Diffusion + RLCD)
    print("--> [1/3] Pre-computing Tier 1 predictions (ModernBERT Diffusion + RLCD)...")
    t1_engine = create_engine("diffusion-rlcd")
    t1_results = {}
    for s in samples:
        d = t1_engine.predict(s["prompt"])
        t1_results[s["id"]] = d

    # 2. Evaluate or load cached System 2 (Llama-Guard-3-1B)
    t2_cache_file = ROOT_DIR / tier2_cache_path
    t2_results = {}

    if t2_cache_file.exists():
        print(f"--> [2/3] Loading cached Tier 2 predictions from {tier2_cache_path}...")
        with open(t2_cache_file, "r", encoding="utf-8") as f:
            t2_data = json.load(f)
        for item in t2_data["detailed_predictions"]:
            t2_results[item["id"]] = item
    else:
        print(f"--> [2/3] Evaluating Tier 2 (Llama-Guard-3-1B on Port 5817) for {n_samples} samples...")
        t2_client = Tier2Client(simulate_delay=False)
        detailed_t2 = []

        for idx, s in enumerate(samples, 1):
            prompt = s["prompt"]
            d2 = await t2_client.evaluate_prompt(prompt)
            pred_is_unsafe = (d2.category != DecisionCategory.SAFE)
            record = {
                "id": s["id"],
                "predicted_verdict": "UNSAFE" if pred_is_unsafe else "SAFE",
                "predicted_category": d2.category.value,
                "confidence": d2.confidence,
                "latency_ms": round(d2.latency_ms, 2),
                "is_correct": (pred_is_unsafe == s["is_malicious"]),
            }
            detailed_t2.append(record)
            t2_results[s["id"]] = record

            if idx % 50 == 0 or idx == n_samples:
                print(f"   [{idx:03d}/{n_samples:03d}] Evaluated System 2...")

        await t2_client.close()

        # Cache results
        t2_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(t2_cache_file, "w", encoding="utf-8") as f:
            json.dump({"total": n_samples, "detailed_predictions": detailed_t2}, f, indent=2)
        print(f"   [OK] Cached Tier 2 evaluations to: {tier2_cache_path}")

    # 3. Sweep Dual-Threshold Grid
    print("--> [3/3] Sweeping dual thresholds (tau_safe, tau_unsafe) across grid...")
    threshold_values = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    records = []
    best_candidate = None
    min_dist_to_ideal = float("inf")

    for tau_s in threshold_values:
        for tau_u in threshold_values:
            router = DecisionRouter(tau_safe=tau_s, tau_unsafe=tau_u)

            tp = 0  # Unsafe correctly blocked
            fp = 0  # Safe incorrectly blocked
            tn = 0  # Safe passed
            fn = 0  # Unsafe passed
            escalations = 0
            latencies = []

            for s in samples:
                sid = s["id"]
                is_malicious = s["is_malicious"]
                t1_d = t1_results[sid]
                action = router.evaluate(t1_d)

                if action == RoutingAction.BYPASS:
                    # Fast Safe
                    is_blocked = False
                    lat = t1_d.latency_ms
                elif action == RoutingAction.BLOCK:
                    # Fast Block
                    is_blocked = True
                    lat = t1_d.latency_ms
                else:
                    # Escalate to Tier 2
                    escalations += 1
                    t2_res = t2_results[sid]
                    is_blocked = (t2_res["predicted_verdict"] == "UNSAFE")
                    lat = t1_d.latency_ms + t2_res["latency_ms"]

                latencies.append(lat)

                if is_malicious:
                    if is_blocked:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if is_blocked:
                        fp += 1
                    else:
                        tn += 1

            dsr = (tp / total_malicious * 100) if total_malicious > 0 else 0.0
            fpr = (fp / total_benign * 100) if total_benign > 0 else 0.0
            acc = (tp + tn) / n_samples * 100
            esc_rate = (escalations / n_samples * 100)
            cost_savings = 100.0 - esc_rate
            mean_lat = float(np.mean(latencies))
            p50_lat = float(np.percentile(latencies, 50))
            p95_lat = float(np.percentile(latencies, 95))

            row = {
                "tau_safe": tau_s,
                "tau_unsafe": tau_u,
                "escalation_rate_pct": round(esc_rate, 2),
                "cost_savings_pct": round(cost_savings, 2),
                "defense_success_rate_pct": round(dsr, 2),
                "false_positive_rate_pct": round(fpr, 2),
                "accuracy_pct": round(acc, 2),
                "mean_latency_ms": round(mean_lat, 2),
                "p50_latency_ms": round(p50_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "escalated_count": escalations,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
            records.append(row)

            # Check for optimal point (e.g. DSR high while escalation low)
            # Distance to ideal (DSR=100%, EscRate=0%)
            dist = np.sqrt(((100.0 - dsr) / 10.0) ** 2 + (esc_rate / 10.0) ** 2)
            if dist < min_dist_to_ideal:
                min_dist_to_ideal = dist
                best_candidate = row

    # Save to JSON
    out_json = ROOT_DIR / output_json_path
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Save to CSV
    out_csv = ROOT_DIR / output_csv_path
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"--> [OK] Saved {len(records)} Pareto grid combinations to {output_json_path} and {output_csv_path}")

    # Plot Pareto Frontiers with matplotlib
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # 1. Trade-off: DSR vs Escalation Rate
        # Group by tau_safe for distinct curves
        for ts in [0.60, 0.70, 0.80, 0.85, 0.90]:
            sub = [r for r in records if r["tau_safe"] == ts]
            sub.sort(key=lambda x: x["escalation_rate_pct"])
            x_esc = [r["escalation_rate_pct"] for r in sub]
            y_dsr = [r["defense_success_rate_pct"] for r in sub]
            ax1.plot(x_esc, y_dsr, marker="o", label=f"$\\tau_{{safe}}={ts:.2f}$", alpha=0.8)

        # Highlight best point
        if best_candidate:
            ax1.scatter(
                [best_candidate["escalation_rate_pct"]],
                [best_candidate["defense_success_rate_pct"]],
                color="red",
                s=120,
                zorder=5,
                label=f"Optimal $(\\tau_s^*={best_candidate['tau_safe']}, \\tau_u^*={best_candidate['tau_unsafe']})$",
            )

        ax1.set_title("Pareto Frontier: Security (DSR %) vs Escalation Rate (%)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Escalation Rate to Tier 2 (%)", fontsize=10)
        ax1.set_ylabel("Defense Success Rate (DSR %)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(fontsize=9, loc="lower right")

        # 2. Cost Savings vs Mean Latency
        for tu in [0.60, 0.70, 0.80, 0.85, 0.90]:
            sub = [r for r in records if r["tau_unsafe"] == tu]
            sub.sort(key=lambda x: x["mean_latency_ms"])
            x_lat = [r["mean_latency_ms"] for r in sub]
            y_sav = [r["cost_savings_pct"] for r in sub]
            ax2.plot(x_lat, y_sav, marker="s", label=f"$\\tau_{{unsafe}}={tu:.2f}$", alpha=0.8)

        ax2.set_title("Compute Cost Savings (%) vs End-to-End Latency (ms)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Mean Latency (ms)", fontsize=10)
        ax2.set_ylabel("Compute Cost Savings vs 100% Tier 2 (%)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(fontsize=9, loc="upper right")

        fig.tight_layout()
        out_plot = ROOT_DIR / output_plot_path
        fig.savefig(out_plot, dpi=200)
        print(f"--> [OK] Exported publication Pareto diagram to: {out_plot}")
    except Exception as e:
        print(f"Notice: Matplotlib plotting note: {e}")

    print("\n" + "=" * 70)
    print(" KẾT QUẢ TỐI ƯU HÓA NGƯỠNG ĐỊNH TUYẾN PARETO FRONTIER")
    print("=" * 70)
    if best_candidate:
        print(f"* Cặp ngưỡng khuyến nghị tối ưu: tau_safe* = {best_candidate['tau_safe']:.2f}, tau_unsafe* = {best_candidate['tau_unsafe']:.2f}")
        print(f"* Tỷ lệ leo thang (Escalation Rate): {best_candidate['escalation_rate_pct']:.2f}% (Tiết kiệm {best_candidate['cost_savings_pct']:.2f}% chi phí tính toán)")
        print(f"* Tỷ lệ phát hiện độc hại (DSR): {best_candidate['defense_success_rate_pct']:.2f}%")
        print(f"* Tỷ lệ dương tính giả (FPR): {best_candidate['false_positive_rate_pct']:.2f}%")
        print(f"* Độ chính xác tổng thể: {best_candidate['accuracy_pct']:.2f}%")
        print(f"* Độ trễ trung bình toàn trình: {best_candidate['mean_latency_ms']:.2f} ms (P50: {best_candidate['p50_latency_ms']:.2f} ms, P95: {best_candidate['p95_latency_ms']:.2f} ms)")
    print("=" * 70)

    return best_candidate


if __name__ == "__main__":
    asyncio.run(sweep_pareto_kg_route())
