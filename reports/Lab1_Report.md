# Báo cáo Lab 1 — Phân loại FashionMNIST bằng MLP

## 1. Mục tiêu

Xây dựng pipeline PyTorch phân loại ảnh thời trang thành 10 lớp:
chuẩn bị dữ liệu, huấn luyện, chọn cấu hình bằng validation,
đánh giá test và kiểm chứng lưu–tải checkpoint.

## 2. Dữ liệu

FashionMNIST gồm 60.000 ảnh train gốc và 10.000 ảnh test.
Train gốc được chia thành 54.000 train và 6.000 validation,
dùng seed 42 và một file split chung cho mọi run.

Ảnh [1,28,28] được chuyển thành float32 trong khoảng 0–1
bằng ToTensor. Nhãn là số nguyên 0–9. Không dùng augmentation.

Chi tiết: [Dữ liệu — TV1](sections/01_data.md).

## 3. Mô hình và huấn luyện

Kiến trúc: Flatten → Linear(784,H) → ReLU → Linear(H,10).
Đầu ra là logits, không thêm Softmax trước CrossEntropyLoss.

H=128 có 101.770 tham số; H=256 có 203.530 tham số.
Mỗi run dùng Adam, batch size 64 và 10 epoch trên CPU.
Seed được đặt lại; model và optimizer được tạo mới mỗi run.

Chi tiết:
- [Mô hình — TV2](sections/02_model.md).
- [Huấn luyện — TV3](sections/03_training.md).

## 4. So sánh và chọn mô hình

Nguồn: outputs/experiment_results.csv.

| Run | H | Learning rate | Best epoch | Val accuracy (%) | Val loss |
|---|---:|---:|---:|---:|---:|
| A | 128 | 0.001 | 10 | 88.4500 | 0.319781 |
| B | 128 | 0.0001 | 10 | 85.9667 | 0.402145 |
| C | 128 | 0.01 | 5 | 87.1667 | 0.368893 |
| D | 256 | 0.001 | 9 | 88.8333 | 0.300137 |

Ưu tiên validation accuracy cao nhất; hòa thì validation loss
thấp hơn. Hòa cả hai thì giữ epoch trước hoặc run xuất hiện trước.

Chọn D tại epoch 9 bằng số liệu đầy đủ, chưa làm tròn.
Không dùng test để lựa chọn và không train thêm sau selection.

Chi tiết: [Thử nghiệm — TV5](sections/05_experiments.md).

## 5. Đánh giá test

Nguồn: outputs/final/test_metrics.json.

- Checkpoint: run D, epoch 9.
- Số ảnh test: 10.000.
- Test accuracy: 88,11%.
- Test loss: 0.32580671586990356.

CSV dự đoán và ảnh minh họa nằm trong outputs/final/.
Loss curves của từng run nằm trong outputs/runs/.

Chi tiết: [Đánh giá — TV4](sections/04_evaluation.md).

## 6. Tích hợp và lưu–tải

13 unittest PASS; smoke một batch PASS.
Reload trên 64 ảnh ghi nhận logits khớp trong dung sai
rtol=1e-5, atol=1e-6 và nhãn dự đoán giống nhau.
Provenance ghi checkpoint nguồn, đích và SHA-256.

Chi tiết: [Tích hợp — TV6](sections/06_integration.md).

## 7. Kết luận và giới hạn

Trong bốn cấu hình khảo sát với một seed và 10 epoch,
D đạt validation accuracy cao nhất, tăng khoảng 0,3833 điểm
phần trăm so với baseline A. Test accuracy cuối đạt 88,11%.

Chưa đánh giá độ biến thiên qua nhiều seed hoặc so sánh với CNN.
Chưa có phân tích định lượng lỗi theo từng lớp.
Checkpoint cuối được đóng gói trong repository. Checkout sạch tại
commit 01f734b đã qua kiểm tra cấu trúc, 13 unittest, đối chiếu
SHA-256, metadata và forward. Việc cài đặt trong môi trường Python
mới chưa được kiểm chứng.
