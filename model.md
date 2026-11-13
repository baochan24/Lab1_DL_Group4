# Mô hình — Lab 1 — FashionMNIST

## 1. Bài toán
Phân loại ảnh thành 10 lớp trên tập **FashionMNIST**. Đầu ra của mạng là 10 logits; lớp có logit lớn nhất được dùng làm dự đoán.

## 2. Kiến trúc và huấn luyện
Flatten (784) → Linear (256) → ReLU → Linear (10). 203.530 tham số. Adam, learning rate 0,001, batch size 64, tối đa 10 epoch; checkpoint ở epoch 9.

## 3. Quy trình
Dùng `src/data.py` → `src/model.py` → `src/train.py` → `src/experiments.py` → `src/evaluate.py`.

## 4. Kết quả mô hình được chọn
- Run: **D**.
- Best validation accuracy: **88.8333%**.
- Best validation loss: **0.300137319**.
- Test accuracy: **88.11%**.
- Tiêu chí chọn: Hàm chọn mô hình ưu tiên validation accuracy cao nhất, nếu bằng nhau xét validation loss thấp hơn. Trường hợp này run D đồng thời có loss nhỏ nhất.

## 5. Lưu trữ mô hình
Có checkpoint `outputs/final/best_model.pth`; SHA-256 có trong `outputs/final/selection_provenance.json`.

## 6. Giới hạn
Thí nghiệm mới chỉ đánh giá 4 cấu hình với một seed. Chưa đối chiếu với kiến trúc CNN hoặc phân tích lỗi chi tiết theo từng lớp.

Các giá trị trên được trích từ `outputs/`; không huấn luyện mô hình khi biên soạn tài liệu này.
