# Training – TV3

## 1. Mục tiêu

TV3 phụ trách xây dựng quy trình huấn luyện mô hình MLP cho bài toán phân loại ảnh FashionMNIST. Quy trình bao gồm huấn luyện theo từng epoch, tính toán loss và accuracy, đánh giá trên tập validation, lưu lịch sử huấn luyện và lựa chọn checkpoint tốt nhất.

## 2. Phương pháp huấn luyện

Mô hình được huấn luyện với hàm mất mát CrossEntropyLoss và bộ tối ưu Adam. Cấu hình A sử dụng learning rate 0.001, 128 neuron ở lớp ẩn và 10 epochs.

Trong mỗi epoch, mô hình được chuyển sang chế độ train. Ảnh và nhãn được đưa lên thiết bị tính toán đã chọn, sau đó thực hiện dự đoán, tính loss, lan truyền ngược và cập nhật trọng số.

Sau mỗi epoch, mô hình được đánh giá trên tập validation. Loss và accuracy được tính theo tổng số mẫu để xử lý chính xác cả batch cuối có kích thước nhỏ hơn.

## 3. Tiêu chí lựa chọn mô hình

Checkpoint tốt nhất được lựa chọn theo accuracy trên tập validation. Accuracy cao hơn được ưu tiên; nếu accuracy bằng nhau thì chọn checkpoint có validation loss thấp hơn. Nếu cả hai chỉ số bằng nhau, giữ checkpoint được chọn trước đó.

Tập test không được sử dụng để lựa chọn checkpoint, nhằm hạn chế rò rỉ thông tin từ dữ liệu kiểm thử.

## 4. Kết quả thực nghiệm

Bốn cấu hình A, B, C và D được huấn luyện trong 10 epochs trên CPU.

| Cấu hình | Best epoch | Validation loss | Validation accuracy |
|---|---:|---:|---:|
| A | 10 | 0.3185 | 88.73% |
| B | 10 | 0.4021 | 85.97% |
| C | 5 | 0.3691 | 87.05% |
| D | 8 | 0.3070 | 88.72% |

Cấu hình A được chọn vì đạt validation accuracy cao nhất, khoảng 88.73%. Cấu hình D có validation loss thấp hơn nhưng accuracy thấp hơn A một lượng nhỏ, vì vậy A được chọn theo tiêu chí đã định.

## 5. Đánh giá trên tập test

Checkpoint được chọn từ cấu hình A, tại epoch 10, được đánh giá trên 10.000 ảnh của tập test FashionMNIST.

Kết quả thực tế:

- Test loss: 0.3409.
- Test accuracy: 87.89%.
- Số lượng ảnh kiểm thử: 10.000.

Kết quả cho thấy mô hình phân loại đúng khoảng 87.89% số ảnh trong tập test. Một số lỗi phân loại xảy ra giữa những lớp có hình dạng tương tự, chẳng hạn Sneaker và Sandal, hoặc Coat và Pullover.

## 6. Các tệp đầu ra

Quy trình huấn luyện tạo ra các tệp chính sau:

- `outputs/runs/A/history.csv`: lịch sử train loss, train accuracy, validation loss và validation accuracy theo epoch.
- `outputs/runs/A/best_model.pth`: checkpoint tốt nhất của cấu hình A.
- `outputs/experiment_results.csv`: bảng tổng hợp kết quả các cấu hình.
- `outputs/final/selection.json`: thông tin lựa chọn mô hình.
- `outputs/final/best_model.pth`: checkpoint được chọn để đánh giá cuối cùng.
- `outputs/final/test_metrics.json`: kết quả đánh giá trên tập test.
- `outputs/final/test_predictions.csv`: nhãn thật và nhãn dự đoán.
- `outputs/final/predictions.png`: hình minh họa các dự đoán.
- `outputs/final/misclassified.png`: hình minh họa các trường hợp phân loại sai.

## 7. Kết luận

Quy trình huấn luyện, lựa chọn checkpoint và đánh giá trên tập test đã chạy thành công. Cấu hình A đạt test accuracy 87.89% trên 10.000 ảnh FashionMNIST.

Việc lưu lịch sử huấn luyện, checkpoint và metadata giúp bảo đảm khả năng kiểm tra lại kết quả. Trong tương lai, có thể thử thêm điều chỉnh learning rate, số neuron lớp ẩn và kỹ thuật regularization để cải thiện khả năng tổng quát hóa của mô hình.
