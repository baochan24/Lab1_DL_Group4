# 04. Đánh giá và trực quan hóa — TV4

## 1. Phương pháp

Chỉ đánh giá tập test sau khi chọn mô hình bằng validation.
Checkpoint cuối là `outputs/final/best_model.pth`, thuộc run D,
epoch 9.

Tập test gồm 10.000 ảnh FashionMNIST, dùng ToTensor, batch size 64,
không shuffle và không bỏ batch cuối. Tên lớp và data root được
lấy từ metadata checkpoint.

Đánh giá dùng model.eval() và torch.no_grad(). Loss được tính
trung bình theo số mẫu, accuracy bằng số dự đoán đúng chia tổng
số mẫu. Không cập nhật trọng số trong bước đánh giá.

## 2. Kết quả thực nghiệm

Nguồn: `outputs/final/test_metrics.json`.

| Chỉ số | Giá trị |
|---|---:|
| Run | D |
| Epoch | 9 |
| Số ảnh test | 10.000 |
| Cross-entropy loss | 0.32580671586990356 |
| Accuracy | 0.8811 — tương đương 88,11% |
| Số dự đoán đúng | 8.811 |
| Số dự đoán sai | 1.189 |

Các số đếm đúng/sai được suy ra từ accuracy và tổng số mẫu.
Validation accuracy của checkpoint được chọn là 88,8333%;
cao hơn test accuracy khoảng 0,7233 điểm phần trăm.

## 3. Output

- `outputs/final/test_metrics.json`: chỉ số đánh giá cuối.
- `outputs/final/test_predictions.csv`: test_index, true_label,
  predicted_label theo đúng thứ tự tập test.
- `outputs/final/predictions.png`: tối đa 12 ảnh với nhãn thật
  và nhãn dự đoán; không đại diện cho toàn bộ tập test.
- `outputs/final/misclassified.png`: tối đa 12 trường hợp
  dự đoán sai thực tế.
- `outputs/runs/<run_id>/loss_curves.png`: train loss và
  validation loss theo epoch của từng run.

## 4. Kiểm tra và giới hạn

Kiểm thử bằng dữ liệu nhỏ đã xác minh loss với batch không đều,
thứ tự nhãn/chỉ số, trọng số không đổi khi đánh giá, đầu ra CSV
và PNG. Kiểm thử dữ liệu nhỏ không thay thế kết quả test thật.

Chưa có phân tích precision/recall/F1 theo lớp trong bộ output
được đối chiếu. Không suy luận loại quần áo dễ nhầm nhất chỉ
từ một số ảnh minh họa.