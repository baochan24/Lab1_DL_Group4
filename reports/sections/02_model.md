# Model — TV2

TODO: viết mục tiêu, cách làm, output thực tế, cách kiểm tra và nhận xét. Không điền kết quả dự kiến thành kết quả đã chạy.

## 1. Mục tiêu

Xây dựng mô hình MLP (Multi-Layer Perceptron) bằng PyTorch để phân loại ảnh FashionMNIST thành 10 lớp. Mô hình nằm trong `src/model.py` và cho phép thay đổi số neuron ở lớp ẩn bằng tham số `hidden_units`.

## 2. dữ liệu theo TV1

Theo `src/data.py` của TV1, dữ liệu được tải bằng `torchvision.datasets.FashionMNIST` với `transforms.ToTensor()`. `DataLoader` trả về một batch ảnh và một batch nhãn.

| Thành phần                 | Quy ước                      |
| -------------------------- | ---------------------------- |
| Một ảnh                    | `[1, 28, 28]`                |
| Batch ảnh với batch size B | `[B, 1, 28, 28]`             |
| Batch nhãn                 | `[B]`                        |
| Kiểu dữ liệu ảnh           | `torch.float32`              |
| Kiểu dữ liệu nhãn          | `torch.int64` / `torch.long` |
| Giá trị nhãn               | Từ 0 đến 9                   |
| Đầu ra mô hình             | `[B, 10]` logits             |

Shape batch thực tế có thể nhỏ hơn 64 ở batch cuối nếu số lượng mẫu không chia hết cho batch size.

## 3. Kiến trúc mô hình

`Input [B, 1, 28, 28] → Flatten → Linear(784, H) → ReLU → Linear(H, 10) → Logits [B, 10]`

- **Flatten:** làm phẳng từng ảnh, không làm phẳng gộp toàn bộ batch; `[B, 1, 28, 28]` thành `[B, 784]`.
- **Linear(784, H):** ánh xạ 784 pixel thành H đặc trưng ẩn.
- **ReLU:** thêm tính phi tuyến.
- **Linear(H, 10):** tạo 10 logits cho mỗi ảnh.

Không đặt Softmax ở cuối mô hình vì `torch.nn.CrossEntropyLoss` nhận logits trực tiếp.

## 4. Số lượng tham số

Tổng số tham số có thể huấn luyện là:

`(784 × H + H) + (H × 10 + 10) = 795H + 10`

| Cấu hình | H (`hidden_units`) | Số tham số |
| -------- | -----------------: | ---------: |
| A        |                128 |    101.770 |
| B        |                128 |    101.770 |
| C        |                128 |    101.770 |
| D        |                256 |    203.530 |

Flatten và ReLU không có tham số cần huấn luyện.

## 5. Kiểm tra

Chạy kiểm thử bằng tensor giả lập trước khi kết nối với DataLoader thực tế:

```python
import torch
from src.model import MLP

model = MLP(hidden_units=128)
images = torch.randn(64, 1, 28, 28)
logits = model(images)

assert logits.shape == (64, 10)
assert torch.isfinite(logits).all()

labels = torch.randint(0, 10, (64,), dtype=torch.long)
loss = torch.nn.CrossEntropyLoss()(logits, labels)
loss.backward()

print("Output shape:", tuple(logits.shape))
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
print("Loss:", loss.item())
print("Backward: OK")
```

Kết quả mong đợi: shape đầu ra `(64, 10)`, số tham số `101770`, loss hữu hạn và `loss.backward()` chạy thành công. Chỉ ghi nhận là đã đạt sau khi thực sự chạy kiểm thử.

## 6. Nhận xét

MLP là mô hình cơ sở đơn giản và dễ thử nghiệm. Tuy nhiên, Flatten không giữ tường minh cấu trúc không gian hai chiều của ảnh như CNN. Đánh giá chất lượng phân loại sẽ được bổ sung sau khi pipeline huấn luyện chạy hoàn chỉnh.
