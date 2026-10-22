# 05. Thử nghiệm và lựa chọn mô hình — TV5

## 1. Thiết kế thí nghiệm

Nhóm khảo sát learning rate và số neuron tầng ẩn của MLP trên FashionMNIST:

| Run | Hidden units | Learning rate |
|---|---:|---:|
| A — baseline | 128 | 0.001 |
| B | 128 | 0.0001 |
| C | 128 | 0.01 |
| D | 256 | 0.001 |

Các run dùng chung split 54.000 train / 6.000 validation, seed 42,
ToTensor, batch size 64, Adam và 10 epoch. Seed được đặt lại trước
khi khởi tạo model mới cho mỗi run. Tập test không tham gia chọn mô hình.

## 2. Kết quả thực nghiệm

Nguồn: `outputs/experiment_results.csv`.
Accuracy trong bảng được chuyển từ thang 0–1 sang phần trăm.

| Run | Best epoch | Validation accuracy (%) | Validation loss |
|---|---:|---:|---:|
| A | 10 | 88.4500 | 0.319781 |
| B | 10 | 85.9667 | 0.402145 |
| C | 5 | 87.1667 | 0.368893 |
| D | 9 | 88.8333 | 0.300137 |

Số liệu được làm tròn khi trình bày; lựa chọn checkpoint sử dụng
giá trị đầy đủ trong output, không dùng số đã làm tròn.

## 3. Nhận xét

- Trong ngân sách 10 epoch, B có validation accuracy thấp nhất.
  Kết quả này chưa chứng minh B sẽ kém hơn nếu huấn luyện lâu hơn.
- C đạt best epoch ở epoch 5, nhưng validation accuracy thấp hơn A.
  Cần xem toàn bộ history/loss curves trước khi kết luận về dao động
  hoặc overfitting.
- D đạt validation accuracy cao nhất và validation loss thấp nhất.
  So với A, accuracy tăng khoảng 0.3833 điểm phần trăm.
- D có nhiều tham số hơn A. Kết quả hiện tại chỉ dựa trên một seed,
  chưa đủ để kết luận mức cải thiện ổn định qua nhiều lần chạy.

## 4. Quy tắc và kết quả lựa chọn

Trong mỗi run, chọn epoch có validation accuracy cao nhất;
nếu hòa, chọn validation loss thấp hơn; nếu tiếp tục hòa,
giữ epoch trước.

Giữa các run, áp dụng cùng quy tắc accuracy/loss; nếu hòa cả hai,
giữ run xuất hiện trước trong danh sách đầu vào.

Kết quả trong `outputs/final/selection.json`:

- Run được chọn: D.
- Best epoch: 9.
- Validation accuracy: 0.8883333333333333.
- Validation loss: 0.3001373190085093.
- Checkpoint nguồn: `outputs/runs/D/best_model.pth`.
- Checkpoint dùng đánh giá cuối: `outputs/final/best_model.pth`.

Không huấn luyện thêm sau bước chọn. Đánh giá test và kiểm chứng
checkpoint được trình bày riêng trong phần TV4 và TV6.