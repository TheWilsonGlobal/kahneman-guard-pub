"""
Standalone Evaluation Script for System 2 (Llama-Guard-3-1B).
Evaluates standalone performance, security metrics (DSR, FPR, Accuracy),
and latency profile on the frozen benchmark kg_eval_cal.jsonl (N=300).
"""

import asyncio
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
from src.core.router import DecisionCategory


async def evaluate_standalone_tier2(
    benchmark_path: str = "data/benchmarks/kg_eval_cal.jsonl",
    output_path: str = "results/tier2_standalone_eval.json",
    limit: int | None = None,
):
    print("=" * 70)
    print(" SYSTEM 2 STANDALONE BENCHMARK: LLAMA-GUARD-3-1B (VULKAN GPU / PORT 5817)")
    print("=" * 70)

    bench_file = ROOT_DIR / benchmark_path
    if not bench_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    samples = []
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if limit is not None:
        samples = samples[:limit]

    print(f"--> Loaded {len(samples)} frozen prompts from {benchmark_path}")

    # Initialize Tier2Client connecting to llama-server on port 5817
    client = Tier2Client(simulate_delay=False)

    detailed_results = []
    latencies = []
    true_positives = 0  # Unsafe correctly identified as Unsafe
    false_positives = 0 # Safe incorrectly blocked as Unsafe
    true_negatives = 0  # Safe correctly identified as Safe
    false_negatives = 0 # Unsafe incorrectly passed as Safe

    total_malicious = sum(1 for s in samples if s["is_malicious"])
    total_benign = sum(1 for s in samples if not s["is_malicious"])

    print(f"--> Starting evaluation (Malicious: {total_malicious}, Benign: {total_benign})...")

    start_total_time = time.perf_counter()

    for idx, sample in enumerate(samples, 1):
        prompt = sample["prompt"]
        is_malicious = sample["is_malicious"]
        ground_truth = sample["ground_truth"]

        decision = await client.evaluate_prompt(prompt)
        pred_is_unsafe = (decision.category != DecisionCategory.SAFE)
        pred_verdict = "UNSAFE" if pred_is_unsafe else "SAFE"
        is_correct = (pred_is_unsafe == is_malicious)

        if is_malicious:
            if pred_is_unsafe:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if pred_is_unsafe:
                false_positives += 1
            else:
                true_negatives += 1

        latencies.append(decision.latency_ms)

        record = {
            "id": sample["id"],
            "prompt": prompt,
            "category": sample.get("category", "Unknown"),
            "source_dataset": sample.get("source_dataset", "Unknown"),
            "ground_truth": ground_truth,
            "is_malicious": is_malicious,
            "predicted_verdict": pred_verdict,
            "predicted_category": decision.category.value,
            "confidence": decision.confidence,
            "is_correct": is_correct,
            "latency_ms": round(decision.latency_ms, 2),
            "details": decision.details,
        }
        detailed_results.append(record)

        if idx % 50 == 0 or idx == len(samples):
            curr_acc = (true_positives + true_negatives) / idx * 100
            curr_mean_lat = np.mean(latencies)
            print(f"   [{idx:03d}/{len(samples):03d}] Current Acc: {curr_acc:.2f}% | Mean Latency: {curr_mean_lat:.1f}ms")

    total_duration = time.perf_counter() - start_total_time
    await client.close()

    # Compute aggregate metrics
    accuracy = (true_positives + true_negatives) / len(samples) * 100
    dsr = (true_positives / total_malicious * 100) if total_malicious > 0 else 0.0
    asr = (false_negatives / total_malicious * 100) if total_malicious > 0 else 0.0
    fpr = (false_positives / total_benign * 100) if total_benign > 0 else 0.0

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    throughput = len(samples) / total_duration

    summary = {
        "model_name": "Llama-Guard-3-1B.Q4_K_M (Vulkan GPU)",
        "benchmark_dataset": benchmark_path,
        "total_samples": len(samples),
        "malicious_samples": total_malicious,
        "benign_samples": total_benign,
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        },
        "metrics": {
            "accuracy_pct": round(accuracy, 2),
            "defense_success_rate_pct (DSR)": round(dsr, 2),
            "attack_success_rate_pct (ASR)": round(asr, 2),
            "false_positive_rate_pct (FPR)": round(fpr, 2),
        },
        "latency_profile_ms": {
            "mean": round(mean_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
            "p50": round(p50_lat, 2),
            "p90": round(p90_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
            "throughput_qps": round(throughput, 2),
            "total_eval_duration_sec": round(total_duration, 2),
        },
        "detailed_predictions": detailed_results,
    }

    out_file = ROOT_DIR / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(" KẾT QUẢ ĐÁNH GIÁ NĂNG LỰC ĐỘC LẬP CỦA SYSTEM 2 (LLAMA GUARD 3-1B)")
    print("=" * 70)
    print(f"* Tổng số mẫu: {len(samples)} (Độc hại: {total_malicious}, An toàn: {total_benign})")
    print(f"* Độ chính xác (Accuracy): {accuracy:.2f}%")
    print(f"* Tỷ lệ chặn độc hại (DSR): {dsr:.2f}% (Chặn được {true_positives}/{total_malicious} ca)")
    print(f"* Tỷ lệ lọt độc hại (ASR): {asr:.2f}% ({false_negatives}/{total_malicious} ca)")
    print(f"* Tỷ lệ dương tính giả (FPR): {fpr:.2f}% ({false_positives}/{total_benign} ca)")
    print("-" * 70)
    print(f"* Độ trễ trung bình (Mean Latency): {mean_lat:.2f} ms")
    print(f"* Phân vị P50: {p50_lat:.2f} ms | P90: {p90_lat:.2f} ms | P95: {p95_lat:.2f} ms")
    print(f"* Thông lượng xử lý (Throughput): {throughput:.2f} queries/giây")
    print(f"* Đã lưu chi tiết vào: {out_file}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    asyncio.run(evaluate_standalone_tier2())
