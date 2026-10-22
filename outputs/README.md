# Kết quả thực nghiệm

Các file ở đây là output của pipeline FashionMNIST đã chạy.

- experiment_results.csv: bảng so sánh A/B/C/D.
- runs/A, B, C, D: config, metadata, history và loss curves.
- final/best_model.pth: checkpoint D, epoch 9.
- final/selection.json: lựa chọn bằng validation.
- final/test_metrics.json: test accuracy 88,11% trên 10.000 ảnh.
- final/test_predictions.csv: nhãn thật và dự đoán theo test_index.
- final/predictions.png, misclassified.png: ảnh minh họa.
- final/reload_check.json: kiểm chứng lưu–tải.
- final/selection_provenance.json: nguồn checkpoint và SHA-256.
- smoke_report.json: kiểm tra tích hợp một batch.

Chỉ checkpoint cuối được đưa vào Git bằng ngoại lệ .gitignore.
Checkpoint riêng của các run không được đóng gói.

Đường dẫn tuyệt đối trong JSON/CSV là thông tin lịch sử của máy
thực hiện. Không cần thư mục gốc đó để nạp checkpoint cuối.

Không ghi đè artifact đã đóng gói khi chạy lại thí nghiệm.
