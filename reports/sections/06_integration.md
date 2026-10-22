# 06. Tích hợp, lưu–tải và tái lập — TV6

## 1. Điều phối

Main đặt seed, chọn device, tạo model và ghi cấu hình từng run.
TV3 thực hiện training; TV4 đánh giá và vẽ; TV5 chọn mô hình bằng
validation. Không dùng test để chọn cấu hình hoặc best epoch.

Main quản lý status/error của vòng đời run. TV3 ghi
training_status/training_error riêng để tránh ghi đè trạng thái
main. Các trạng thái vòng đời gồm running, failed, trained và
completed; lỗi vẽ được ghi riêng.

## 2. Môi trường đã kiểm tra

Theo output doctor trên máy thực hiện:

| Thành phần | Phiên bản/trạng thái |
|---|---|
| Hệ điều hành | Windows 11 |
| Python | 3.13.7 |
| PyTorch | 2.14.1+cpu |
| Torchvision | 0.29.1+cpu |
| NumPy | 2.4.4 |
| Matplotlib | 3.10.9 |
| Pandas | 3.0.6 |
| Device | CPU |
| CUDA khả dụng | Không |

Doctor đã import được các thư viện và thực hiện phép tính tensor.
Kết quả này chỉ xác nhận môi trường, không xác nhận chất lượng model.
Thời gian chạy không được so sánh trực tiếp với máy khác.

## 3. Kiểm thử tích hợp

13/13 unittest đã PASS. Các kiểm thử kiểm tra schema checkpoint,
split, JSON và điều phối trạng thái; các fixture tạm không phải
số liệu FashionMNIST.

`outputs/smoke_report.json` ghi nhận:
- Batch ảnh: [64, 1, 28, 28].
- Device: CPU.
- Trọng số thay đổi sau một bước training.
- Đánh giá không thay đổi trọng số.
- Logits và nhãn khớp sau lưu–tải.
- passed: true.

Smoke chỉ kiểm ghép một batch, không phải validation/test accuracy.

## 4. Checkpoint cuối và provenance

- Checkpoint nguồn: `outputs/runs/D/best_model.pth`.
- Checkpoint cuối: `outputs/final/best_model.pth`.
- Run D, epoch 9.
- Các ứng viên: A, B, C, D.

`outputs/final/selection_provenance.json` ghi SHA-256:

f2cfc2dd397695916d4044bf7233e8143de206e88ded51990d260e0a62803a52

SHA-256 dùng để đối chiếu tính toàn vẹn của file checkpoint.

## 5. Kiểm tra reload

`outputs/final/reload_check.json` ghi nhận:
- Số mẫu: 64.
- logits_match: true.
- labels_match: true.
- rtol: 1e-5; atol: 1e-6.
- passed: true.

Kiểm tra xác nhận dự đoán nhất quán giữa các mô hình được so sánh
trên batch đã dùng. Không suy rộng thành bảo đảm đồng nhất tuyệt
đối giữa mọi phần cứng và phiên bản thư viện.

## 6. Trạng thái tái lập và bàn giao

Pipeline đã được chạy tại worktree tích hợp trên máy thực hiện.
Chưa ghi nhận kiểm chứng từ clone sạch trong các bằng chứng hiện có.

Checkpoint được giữ ngoài Git theo quy tắc *.pth/*.pt.
Kênh lưu bền vững, hướng dẫn lấy checkpoint và kiểm chứng từ
checkout sạch sẽ được hoàn tất trong bước đóng gói bàn giao.