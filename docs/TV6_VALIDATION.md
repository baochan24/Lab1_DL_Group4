# Xác minh TV6 — 20/10/2026

- `python main.py --mode check`: qua trên snapshot main với data.py TV1.
- `python -m compileall -q main.py src tests`: qua.
- `python -m unittest discover -s tests -v`: 13 test qua (schema, dữ liệu split, JSON và orchestration với member functions thay thế trong test).
- `python main.py --mode doctor`: tại môi trường chuẩn bị báo chưa sẵn sàng vì thiếu torch/torchvision; đây là kết quả kiểm tra trung thực, không là lỗi thuật toán.
- Không chạy FashionMNIST training, evaluate cuối hoặc model reload thật; chờ môi trường và các phần còn TODO.

Unit tests dùng thư mục tạm và dữ liệu sentinel để kiểm lifecycle; không lưu chúng vào outputs/reports như kết quả ML. Không tạo requirements-lock giả.
