# 05. Thử nghiệm và Lựa chọn Mô hình (TV5)

## 1. Thiết kế Thử nghiệm
Để khảo sát ảnh hưởng của các siêu tham số đến hiệu năng mạng MLP trên bộ dữ liệu FashionMNIST, nhóm tiến hành 4 đợt chạy thử nghiệm (A, B, C, D):

- **Run A (Baseline):** 128 hidden units, learning rate = 0.001.
- **Run B (LR nhỏ):** 128 hidden units, learning rate = 0.0001 (đánh giá tốc độ hội tụ khi giảm LR).
- **Run C (LR lớn):** 128 hidden units, learning rate = 0.01 (đánh giá sự ổn định khi tăng LR).
- **Run D (Mạng rộng hơn):** 256 hidden units, learning rate = 0.001 (đánh giá khả năng biểu diễn khi tăng dung lượng mô hình).

Tất cả các thử nghiệm đều tuân thủ các quy tắc kiểm soát biến số:
- Sử dụng chung một bộ chỉ số phân chia dữ liệu (`outputs/split_indices.json`).
- Đặt lại seed khởi tạo ngẫu nhiên cố định (`seed = 42`).
- Cùng huấn luyện trong 10 epochs với `batch_size = 64`.

## 2. Kết quả So sánh

Sau khi hoàn tất quá trình huấn luyện, kết quả được tổng hợp tự động vào `outputs/experiment_results.csv`:

| Run ID | Hidden Units | Learning Rate | Best Epoch | Best Val Accuracy | Best Val Loss |
|---|---:|---:|---:|---:|---:|
| **A** | 128 | 0.001 | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* |
| **B** | 128 | 0.0001 | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* |
| **C** | 128 | 0.01 | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* |
| **D** | 256 | 0.001 | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* | *[Ghi nhận khi train]* |

## 3. Phân tích và Đánh giá

- **Tác động của Learning Rate (A vs B vs C):**
  - Learning rate quá nhỏ (Run B) giúp quá trình tối ưu ổn định nhưng tốc độ hội tụ chậm, có thể cần thêm epoch để đạt hiệu năng tối ưu.
  - Learning rate quá lớn (Run C) làm bước cập nhật trọng số lớn, dễ gây dao động loss hoặc nhảy qua điểm tối ưu cục bộ.
- **Tác động của Kích thước Tầng ẩn (A vs D):**
  - Tăng số neuron từ 128 lên 256 (Run D) mở rộng dung lượng biểu diễn của mạng MLP, giúp học các đặc trưng phức tạp hơn nhưng làm tăng chi phí tính toán và số lượng tham số.

## 4. Quyết định Chọn Mô hình

Quy trình tự động lựa chọn checkpoint tốt nhất được thực hiện qua hàm `summarize_and_select` theo các nguyên tắc nghiêm ngặt:
1. **Tuyệt đối không dùng tập Test** để tham gia vào quá trình so sánh và lựa chọn cấu hình.
2. Ưu tiên chọn run có `val_accuracy` tốt nhất ở epoch tối ưu.
3. Nếu `val_accuracy` bằng nhau, chọn run có `val_loss` thấp hơn. Nếu tiếp tục hòa, chọn run xuất hiện trước trong danh sách truyền vào.

Kết quả được tự động ghi nhận vào file `outputs/final/selection.json` để phục vụ đánh giá tập test (TV4) và kiểm tra reload (TV6).# experiments — TV5

TODO: viết mục tiêu, cách làm, output thực tế, cách kiểm tra và nhận xét. Không điền kết quả dự kiến thành kết quả đã chạy.
