"""
Training module for KahnemanGuard System 1 on NVIDIA L4 (Single-GPU Ada Lovelace architecture, 24GB VRAM).
Optimized for:
  - Native Bfloat16 (torch.bfloat16)
  - Gradient Checkpointing & Micro-batch size for zero OOM risk
  - Both Baseline Cross-Entropy (CE) and RLCD (Gaussian perturbation on logits + Brier reward)
  - L-BFGS Temperature Scaling calibration on holdout calibration split
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from safetensors.torch import save_file


class KahnemanDecisionGuard(nn.Module):
    def __init__(self, model_id: str, use_flash_attn: bool = True):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id)
        if use_flash_attn:
            try:
                config._attn_implementation = "flash_attention_2"
            except Exception:
                pass
        self.encoder = AutoModel.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        # Enable gradient checkpointing to slash activation memory by 60%
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        logits = self.classifier(pooled)
        return logits


def collate_fn(items: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(it["input_ids"]) for it in items)
    bs = len(items)
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    targets = torch.zeros((bs, 2), dtype=torch.float32)
    labels = torch.zeros(bs, dtype=torch.long)

    for i, it in enumerate(items):
        l = len(it["input_ids"])
        input_ids[i, :l] = torch.tensor(it["input_ids"], dtype=torch.long)
        attention_mask[i, :l] = 1
        targets[i] = torch.tensor(it["target"], dtype=torch.float32)
        labels[i] = it["label"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "labels": labels
    }


def compute_brier_reward(pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    brier_score = torch.sum((pred_probs - target_probs) ** 2, dim=-1)
    return 1.0 - brier_score


def fit_temperature(logits_list: list[np.ndarray], targets_list: list[np.ndarray]) -> float:
    Z = torch.tensor(np.array(logits_list), dtype=torch.float32)
    T_target = torch.tensor(np.array(targets_list), dtype=torch.float32)
    log_temp = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.08, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(T_target * torch.log_softmax(Z / log_temp.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temp.exp(), 0.1, 5.0).item())


def parse_args():
    parser = argparse.ArgumentParser(description="Train ModernBERT System 1 on NVIDIA L4 GPU")
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized train_items.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--mode", type=str, default="rlcd", choices=["ce", "rlcd"], help="Training mode: ce or rlcd")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size (16 for safe VRAM)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation (16x2 = 32 effective BS)")
    parser.add_argument("--group_size", type=int, default=4, help="RLCD exploration samples per prompt")
    parser.add_argument("--lr_encoder", type=float, default=2e-5, help="Learning rate for ModernBERT encoder")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for classifier head")
    return parser.parse_args()


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA device required. Make sure Colab L4 is connected."
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n==================================================================")
    print(f"  KahnemanGuard L4 Trainer: Mode={args.mode.upper()}")
    print(f"  Device: {gpu_name} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    print(f"  Effective Batch Size: {args.batch_size * args.grad_accum} (Micro: {args.batch_size}, Accum: {args.grad_accum})")
    print(f"  Gradient Checkpointing: ENABLED (Zero OOM)")
    print(f"==================================================================\n")

    model_id = "answerdotai/ModernBERT-large"
    tok = AutoTokenizer.from_pretrained(model_id)

    model = KahnemanDecisionGuard(model_id, use_flash_attn=False)
    model.to(device)
    model.train()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} not found.")

    all_items = torch.load(args.data_path, weights_only=False)
    random.Random(42).shuffle(all_items)

    n_calib = min(500, max(200, int(len(all_items) * 0.05)))
    calib_items = all_items[:n_calib]
    train_items = all_items[n_calib:]

    print(f"--> Dataset partitioned: {len(train_items)} Train | {len(calib_items)} Calibration Holdout")

    enc_params = [p for n, p in model.named_parameters() if "encoder" in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=0.01)

    total_updates = (len(train_items) // (args.batch_size * args.grad_accum)) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    SIGMA_START = 0.35
    SIGMA_END = 0.05

    t0 = time.time()
    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        random.shuffle(train_items)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_batches = 0

        progress = epoch / max(1, args.epochs - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(train_items), args.batch_size):
            chunk = train_items[b_idx : b_idx + args.batch_size]
            if not chunk:
                continue

            batch = collate_fn(chunk, tok.pad_token_id)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_probs = batch["targets"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=dtype):
                base_logits = model(input_ids, attention_mask)

                if args.mode == "ce":
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(base_logits, labels)
                else:
                    bs = base_logits.shape[0]
                    G = args.group_size
                    expanded_logits = base_logits.unsqueeze(1).repeat(1, G, 1)
                    noise = torch.randn_like(expanded_logits) * sigma
                    perturbed_logits = expanded_logits + noise

                    cand_probs = torch.softmax(perturbed_logits, dim=-1)
                    targets_expanded = target_probs.unsqueeze(1).repeat(1, G, 1)
                    cand_rewards = compute_brier_reward(cand_probs, targets_expanded)

                    mean_r = cand_rewards.mean(dim=-1, keepdim=True)
                    std_r = cand_rewards.std(dim=-1, keepdim=True) + 1e-6
                    adv = (cand_rewards - mean_r) / std_r

                    best_idx = torch.argmax(cand_rewards, dim=-1)
                    teacher_probs = cand_probs[torch.arange(bs, device=device), best_idx].detach()

                    log_p = torch.log_softmax(base_logits, dim=-1)
                    kl_loss = -(teacher_probs * log_p).sum(dim=-1).mean()
                    ce_loss = -(target_probs * log_p).sum(dim=-1).mean()
                    rl_drift_loss = -(adv.detach() * cand_rewards).mean()

                    loss = 0.5 * ce_loss + 0.3 * kl_loss + 0.2 * rl_drift_loss

            loss = loss / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1

            if n_batches % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if n_batches % 100 == 0:
                cur_loss = epoch_loss / n_batches
                print(f"  [Epoch {epoch+1}/{args.epochs}] Step {n_batches}/{len(train_items)//args.batch_size} | Loss: {cur_loss:.4f} | VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"--> [Epoch {epoch+1}/{args.epochs} COMPLETED] Avg Loss: {avg_loss:.4f} | Total Time: {elapsed:.1f}s\n")

    # Calibration Temperature Fitting via L-BFGS on holdout set
    print("--> Fitting calibration temperature T via L-BFGS on holdout calibration split...")
    model.eval()
    calib_logits, calib_targets = [], []
    with torch.no_grad():
        for i in range(0, len(calib_items), args.batch_size):
            c_chunk = calib_items[i : i + args.batch_size]
            cb = collate_fn(c_chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=dtype):
                out = model(cb["input_ids"].to(device), cb["attention_mask"].to(device))
            calib_logits.extend(out.float().cpu().numpy())
            calib_targets.extend(cb["targets"].numpy())

    fitted_temp = fit_temperature(calib_logits, calib_targets)
    print(f"--> Fitted Calibration Temperature T: {fitted_temp:.4f}")

    # Save output artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(args.output_dir, "model.safetensors"))
    tok.save_pretrained(args.output_dir)

    cfg = {
        "model_id": model_id,
        "mode": args.mode,
        "gpu_device": "NVIDIA L4 (24GB VRAM)",
        "temperature": fitted_temp,
        "tau_confidence": 0.80,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(args.output_dir, "guardrail_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[DONE] Checkpoint & Config saved successfully to {args.output_dir}")


if __name__ == "__main__":
    main()
