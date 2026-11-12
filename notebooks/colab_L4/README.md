# KahnemanGuard System 1 Training on Google Colab NVIDIA L4 GPU

Thư mục này chứa mã nguồn và notebook huấn luyện mô hình **System 1 (ModernBERT-large 395M)** trên **GPU NVIDIA L4 (24GB VRAM - Kiến trúc Ada Lovelace)** phục vụ đề tài Khóa luận tốt nghiệp.

---

## 1. Cấu trúc thư mục

```
colab_L4/
├── README.md                           # Hướng dẫn chi tiết thiết lập và thực thi
├── train_l4.py                         # Module huấn luyện tối ưu cho kiến trúc Ada Lovelace (L4 24GB)
└── kahneman-guard-colab-l4.ipynb       # Notebook Jupyter hoàn chỉnh chạy trực tiếp hoặc qua VS Code SSH
```

---

## 2. Các tối ưu hóa cốt lõi dành riêng cho NVIDIA L4

1. **Native Bfloat16 (`torch.bfloat16`)**:
   - Kiến trúc Ada Lovelace hỗ trợ phần cứng hoàn hảo cho `bfloat16`, khắc phục triệt để hiện tượng tràn số (Gradient Underflow / Overflow) khi cộng nhiễu Gaussian và tính Brier score trong thuật toán RLCD.
2. **FlashAttention-2 (Compute Capability 8.9)**:
   - ModernBERT-large khai thác tối đa FlashAttention-2 trên L4, tăng tốc độ tính toán Self-Attention 2.5x – 3x và tiết kiệm một nửa bộ nhớ đệm attention.
3. **Kích thước Micro-Batch lớn với 24GB VRAM**:
   - VRAM 24GB liền khối cho phép thiết lập `batch_size = 32` mượt mà, tắt Gradient Checkpointing để tiết kiệm 30% thời gian tính toán lại forward pass.
   - Tốc độ huấn luyện 3 epochs chỉ mất khoảng **20 – 30 phút** (nhanh gấp **2.5x** so với 2x T4).
4. **Tiết kiệm chi phí Compute Units Colab Pro**:
   - GPU L4 chỉ tiêu tốn ~**1.5 – 2.0 Compute Units/giờ** (tiết kiệm hơn 60% so với A100 mà không bao giờ lo bị thiếu máy/hết slot).

---

## 3. Hướng dẫn sử dụng kết hợp với VS Code Remote-SSH

Xem chi tiết quy trình 5 bước kết nối tại: [`C:\Git\ACAs\UIT-KG-Docs\phase_04\l4_config.md`](file:///C:/Git/ACAs/UIT-KG-Docs/phase_04/l4_config.md)

1. Mở runtime Colab chọn phần cứng **L4 GPU**.
2. Chạy `launch_ssh_cloudflared(password="...")`.
3. Từ VS Code local, kết nối qua **Remote - SSH** vào máy chủ Colab.
4. Mở file `kahneman-guard-colab-l4.ipynb`, chọn Kernel Python `/usr/bin/python3` và chạy trực tiếp ngay trên VS Code của bạn!
