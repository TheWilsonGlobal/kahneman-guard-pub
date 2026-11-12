import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer, AutoConfig
from safetensors.torch import save_file

class KahnemanDecisionGuard(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id)
        self.encoder = AutoModel.from_pretrained(model_id, config=config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] representation
        logits = self.classifier(pooled)
        return logits

def collate_fn(items, pad_id):
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

def compute_brier_reward(pred_probs, target_probs):
    brier_score = torch.sum((pred_probs - target_probs) ** 2, dim=-1)
    return 1.0 - brier_score

def fit_temperature(logits_list, targets_list):
    Z = torch.tensor(np.array(logits_list), dtype=torch.float32)
    T = torch.tensor(np.array(targets_list), dtype=torch.float32)
    log_temp = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_temp.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temp.exp(), 0.1, 5.0).item())

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    model_id = "answerdotai/ModernBERT-large"

    if not os.path.exists(data_path):
        if rank == 0:
            print(f"\n[ERROR] {data_path} not found.")
        dist.destroy_process_group()
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = KahnemanDecisionGuard(model_id)
    model.encoder.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    all_items = torch.load(data_path, weights_only=False)

    random.Random(42).shuffle(all_items)
    n_calib = 500
    calib_items = all_items[:n_calib]
    train_items = all_items[n_calib:]
    my_items = train_items[rank::world_size]

    EPOCHS = 3
    MICRO_BATCH = 8
    GRAD_ACCUM = 4
    GROUP_SIZE = 4
    SIGMA_START = 0.35
    SIGMA_END = 0.05

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder" in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "classifier" in n]

    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": 1.5e-5},
        {"params": head_params, "lr": 1.0e-4}
    ], weight_decay=0.01)

    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    if rank == 0:
        print(f"Starting Multi-GPU RLCD: {len(train_items)} train items | {len(calib_items)} calib | {EPOCHS} epochs")

    t0 = time.time()
    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        accum_step = 0
        optimizer.zero_grad(set_to_none=True)

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_fn(chunk, tok.pad_token_id)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            with torch.autocast("cuda", dtype=torch.float16):
                logits = ddp_model(input_ids, attention_mask).float()

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma
            eps = eps - eps.mean(dim=-1, keepdim=True)
            noisy_z = logits.detach().unsqueeze(0) + eps
            noisy_q = torch.softmax(noisy_z, dim=-1)

            with torch.no_grad():
                rewards = compute_brier_reward(noisy_q, targets.unsqueeze(0))
                adv = rewards - rewards.mean(dim=0, keepdim=True)
                adv = adv / (adv.std(dim=0, keepdim=True) + 1e-6)

            logp = -((noisy_z - logits.unsqueeze(0)) ** 2).sum(-1) / (2 * (sigma ** 2))
            loss_rl = -(adv * logp).mean()
            loss_ce = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if rank == 0:
            print(f"=== Epoch {epoch+1}/{EPOCHS} complete in {time.time()-t0:.1f}s ===")
        dist.barrier()

    if rank == 0:
        print("\n--> Fitting calibration temperature via L-BFGS on holdout set...")
        model.eval()
        calib_logits, calib_targets = [], []
        with torch.no_grad():
            for i in range(0, len(calib_items), 16):
                c_chunk = calib_items[i : i + 16]
                cb = collate_fn(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(cb["input_ids"].to(device), cb["attention_mask"].to(device))
                calib_logits.extend(out.float().cpu().numpy())
                calib_targets.extend(cb["targets"].numpy())

        fitted_temp = fit_temperature(calib_logits, calib_targets)
        print(f"--> Fitted Calibration Temperature: {fitted_temp:.4f}")

        os.makedirs(output_dir, exist_ok=True)
        save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
        tok.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "guardrail_config.json"), "w") as f:
            json.dump({
                "model_id": model_id,
                "temperature": fitted_temp,
                "tau_confidence": 0.80
            }, f, indent=2)
        print(f"--> Final KahnemanGuard model saved to {output_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()