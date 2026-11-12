"""
Offline Synergy Analysis & Net Correction Gain (NCG) Calculator.
Matches System 1 (RLCD Calibrated) and System 2 (Llama-Guard-3-1B) predictions
on the frozen benchmark kg_eval_cal.jsonl (N=300).
Constructs 4-quadrant error matrix and computes Net Correction Gain (NCG).
"""

import csv
import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from src.core.router import DecisionCategory
from src.engines import create_engine


def analyze_synergy(
    tier2_results_path: str = "results/tier2_standalone_eval.json",
    benchmark_path: str = "data/benchmarks/kg_eval_cal.jsonl",
    output_csv_path: str = "results/offline_synergy_matrix.csv",
    output_summary_path: str = "results/offline_synergy_summary.json",
):
    print("=" * 70)
    print(" OFFLINE SYNERGY ANALYSIS & NET CORRECTION GAIN (NCG)")
    print("=" * 70)

    # 1. Load Tier 2 Results
    t2_file = ROOT_DIR / tier2_results_path
    if not t2_file.exists():
        raise FileNotFoundError(f"Tier 2 evaluation file not found: {t2_file}. Run scripts/eval_tier2_standalone.py first.")

    with open(t2_file, "r", encoding="utf-8") as f:
        t2_data = json.load(f)

    t2_preds = {item["id"]: item for item in t2_data["detailed_predictions"]}
    print(f"--> Loaded {len(t2_preds)} Tier 2 predictions from {tier2_results_path}")

    # 2. Load Benchmark samples and run Tier 1 RLCD Engine
    bench_file = ROOT_DIR / benchmark_path
    samples = []
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"--> Initializing Tier 1 (ModernBERT Diffusion + RLCD Engine)...")
    t1_engine = create_engine("diffusion-rlcd")

    synergy_records = []
    both_correct = 0
    both_wrong = 0
    s1_correct_s2_wrong = 0
    s2_corrects_s1 = 0

    s1_correct_count = 0
    s2_correct_count = 0

    print("--> Computing cross-tier decisions for all prompts...")
    for sample in samples:
        sid = sample["id"]
        prompt = sample["prompt"]
        is_malicious = sample["is_malicious"]
        ground_truth = sample["ground_truth"]

        # Run Tier 1
        t1_dec = t1_engine.predict(prompt)
        p_harm = t1_dec.details.get("p_harm", 0.95 if t1_dec.category != DecisionCategory.SAFE else 0.05)
        p_safe = t1_dec.details.get("p_safe", 1.0 - p_harm)
        s1_is_unsafe = (t1_dec.category != DecisionCategory.SAFE)
        s1_verdict = "UNSAFE" if s1_is_unsafe else "SAFE"
        s1_correct = (s1_is_unsafe == is_malicious)

        # Get Tier 2 prediction
        t2_item = t2_preds.get(sid, {})
        s2_verdict = t2_item.get("predicted_verdict", "SAFE")
        s2_is_unsafe = (s2_verdict == "UNSAFE")
        s2_correct = (s2_is_unsafe == is_malicious)

        if s1_correct:
            s1_correct_count += 1
        if s2_correct:
            s2_correct_count += 1

        # Determine synergy classification
        if s1_correct and s2_correct:
            synergy_group = "Both_Correct"
            both_correct += 1
        elif (not s1_correct) and (not s2_correct):
            synergy_group = "Both_Wrong"
            both_wrong += 1
        elif s1_correct and (not s2_correct):
            synergy_group = "S1_Correct_S2_Wrong"
            s1_correct_s2_wrong += 1
        else: # not s1_correct and s2_correct
            synergy_group = "S2_Corrects_S1"
            s2_corrects_s1 += 1

        synergy_records.append({
            "id": sid,
            "category": sample.get("category", "Unknown"),
            "source_dataset": sample.get("source_dataset", "Unknown"),
            "ground_truth": ground_truth,
            "is_malicious": is_malicious,
            "s1_verdict": s1_verdict,
            "s1_p_harm": round(p_harm, 4),
            "s1_confidence": round(t1_dec.confidence, 4),
            "s1_latency_ms": round(t1_dec.latency_ms, 2),
            "s1_correct": s1_correct,
            "s2_verdict": s2_verdict,
            "s2_confidence": round(t2_item.get("confidence", 0.5), 4),
            "s2_latency_ms": round(t2_item.get("latency_ms", 0.0), 2),
            "s2_correct": s2_correct,
            "synergy_group": synergy_group,
            "prompt": prompt,
        })

    # Save detailed CSV
    out_csv = ROOT_DIR / output_csv_path
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id", "category", "source_dataset", "ground_truth", "is_malicious",
            "s1_verdict", "s1_p_harm", "s1_confidence", "s1_latency_ms", "s1_correct",
            "s2_verdict", "s2_confidence", "s2_latency_ms", "s2_correct",
            "synergy_group", "prompt",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(synergy_records)

    total_n = len(synergy_records)
    # Global Net Correction Gain across all 300 samples
    global_ncg = (s2_corrects_s1 - s1_correct_s2_wrong) / total_n

    # Analyze Escalation Zone Sensitivity under confidence thresholding
    # Sweep tau_safe in [0.60, 0.85] and tau_unsafe in [0.60, 0.85]
    routing_synergy_analysis = []
    tau_pairs = [
        (0.60, 0.60),
        (0.70, 0.70),
        (0.75, 0.75),
        (0.80, 0.80),
        (0.85, 0.85),
        (0.70, 0.85),
        (0.65, 0.80),
    ]

    for tau_s, tau_u in tau_pairs:
        # A prompt escalates if tau_s <= p_harm <= tau_u, OR if confidence < threshold
        # In dual-threshold:
        # p_harm < tau_s -> S1 decides SAFE
        # p_harm > tau_u -> S1 decides UNSAFE
        # tau_s <= p_harm <= tau_u -> ESCALATE to S2
        escalated_samples = [
            r for r in synergy_records
            if tau_s <= r["s1_p_harm"] <= tau_u
        ]
        n_esc = len(escalated_samples)
        if n_esc > 0:
            esc_s2_corrects_s1 = sum(1 for r in escalated_samples if r["synergy_group"] == "S2_Corrects_S1")
            esc_s1_correct_s2_wrong = sum(1 for r in escalated_samples if r["synergy_group"] == "S1_Correct_S2_Wrong")
            esc_ncg = (esc_s2_corrects_s1 - esc_s1_correct_s2_wrong) / n_esc
            esc_rate = n_esc / total_n * 100
        else:
            esc_s2_corrects_s1 = 0
            esc_s1_correct_s2_wrong = 0
            esc_ncg = 0.0
            esc_rate = 0.0

        routing_synergy_analysis.append({
            "tau_safe": tau_s,
            "tau_unsafe": tau_u,
            "n_escalated": n_esc,
            "escalation_rate_pct": round(esc_rate, 2),
            "s2_corrects_s1_count": esc_s2_corrects_s1,
            "s1_correct_s2_wrong_count": esc_s1_correct_s2_wrong,
            "ncg": round(esc_ncg, 4),
        })

    summary = {
        "benchmark": benchmark_path,
        "total_samples": total_n,
        "system_1_accuracy_pct": round(s1_correct_count / total_n * 100, 2),
        "system_2_accuracy_pct": round(s2_correct_count / total_n * 100, 2),
        "error_matrix_4_quadrants": {
            "group_1_both_correct": {
                "count": both_correct,
                "pct": round(both_correct / total_n * 100, 2),
                "description": "Cả hai hệ thống đều phân loại đúng (S1 Đúng & S2 Đúng)"
            },
            "group_2_both_wrong": {
                "count": both_wrong,
                "pct": round(both_wrong / total_n * 100, 2),
                "description": "Cả hai hệ thống đều phân loại sai (S1 Sai & S2 Sai)"
            },
            "group_3_s1_correct_s2_wrong": {
                "count": s1_correct_s2_wrong,
                "pct": round(s1_correct_s2_wrong / total_n * 100, 2),
                "description": "S1 đúng nhưng S2 đoán sai (Penalty)"
            },
            "group_4_s2_corrects_s1": {
                "count": s2_corrects_s1,
                "pct": round(s2_corrects_s1 / total_n * 100, 2),
                "description": "S1 đoán sai và được S2 sửa đúng (Correction)"
            },
        },
        "global_net_correction_gain": round(global_ncg, 4),
        "routing_synergy_analysis": routing_synergy_analysis,
    }

    out_sum = ROOT_DIR / output_summary_path
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(" KẾT QUẢ PHÂN TÍCH MA TRẬN BỔ TRỢ ĐA TẦNG (SYNERGY MATRIX)")
    print("=" * 70)
    print(f"* Độ chính xác System 1 (RLCD): {s1_correct_count / total_n * 100:.2f}% ({s1_correct_count}/{total_n})")
    print(f"* Độ chính xác System 2 (Llama Guard 3): {s2_correct_count / total_n * 100:.2f}% ({s2_correct_count}/{total_n})")
    print("-" * 70)
    print("MA TRẬN LỖI 4 NHÓM:")
    print(f"  [Nhóm 1] Cả hai đều đúng (S1 Đúng & S2 Đúng): {both_correct:3d} ca ({both_correct/total_n*100:5.2f}%)")
    print(f"  [Nhóm 2] Cả hai đều sai (S1 Sai & S2 Sai):     {both_wrong:3d} ca ({both_wrong/total_n*100:5.2f}%)")
    print(f"  [Nhóm 3] S1 đúng, S2 sai (Penalty):           {s1_correct_s2_wrong:3d} ca ({s1_correct_s2_wrong/total_n*100:5.2f}%)")
    print(f"  [Nhóm 4] S1 sai, S2 sửa đúng (Correction):     {s2_corrects_s1:3d} ca ({s2_corrects_s1/total_n*100:5.2f}%)")
    print("-" * 70)
    print(f"* Net Correction Gain Toàn thể: {global_ncg:+.4f}")
    print(f"* Đã lưu chi tiết vào: {out_csv}")
    print(f"* Đã lưu báo cáo JSON: {out_sum}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    analyze_synergy()
