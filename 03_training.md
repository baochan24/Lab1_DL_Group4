# TV3 — Huấn luyện mô hình

## Mục tiêu

TV3 triển khai vòng lặp huấn luyện cho MLP phân loại FashionMNIST. Hàm `train_one_epoch` thực hiện forward, tính CrossEntropyLoss, backpropagation và cập nhật trọng số bằng Adam. Hàm `fit` điều phối các epoch, gọi `evaluate` của TV4 trên validation và lưu lịch sử/checkpoint.

## Quy trình

1. `model.train()` được gọi ở đầu mỗi epoch huấn luyện.
2. Ảnh và nhãn được chuyển đến cùng thiết bị với mô hình. Nhãn có kiểu `long`.
3. Trước mỗi batch, optimizer xóa gradient; sau forward và loss thì gọi `backward()` rồi `step()`.
4. Loss trung bình epoch được tính theo số mẫu: tổng `batch_loss × batch_size` chia tổng số mẫu. Accuracy nằm trong khoảng 0–1.
5. `fit` tạo `CrossEntropyLoss` và Adam một lần cho mỗi run, ghi `history.csv` với các cột `epoch,train_loss,train_accuracy,val_loss,val_accuracy`.
6. Epoch tốt nhất được chọn bằng validation accuracy cao nhất; nếu bằng nhau chọn validation loss thấp nhất; nếu cả hai bằng nhau giữ epoch xuất hiện trước. Test set không được sử dụng trong quá trình chọn epoch.
7. Checkpoint `best_model.pth` chứa `model_state_dict`, `model_config`, `preprocessing`, `class_names`, `run_id`, `epoch` và `val_metrics` theo `docs/TEAM_CONTRACT.md`.
8. `metadata.json` ghi trạng thái `running`, `completed` hoặc `failed`, thời gian bắt đầu/kết thúc, thời lượng, epoch tốt nhất, các validation metrics và lỗi nếu có.

## Đầu ra

Trong `outputs/runs/<run_id>/`, `fit` tạo `history.csv`, `best_model.pth` và cập nhật `metadata.json`. `main.py` gọi phần vẽ đường cong sau khi `fit` kết thúc để tạo `loss_curves.png`.

## Lệnh kiểm tra và chạy

Từ thư mục gốc dự án:

```powershell
python -m py_compile src/data.py src/model.py src/train.py src/evaluate.py src/visualize.py src/experiments.py
python main.py --mode check
python main.py --mode prepare --config configs/A.json
python main.py --mode train --config configs/A.json
```

`prepare` cần tải FashionMNIST lần đầu nên máy cần internet. Khi đã có kết quả cho run A, có thể dùng cùng pipeline cho B/C/D bằng config tương ứng. Không ghi đè run cũ; đổi `run_id` khi chạy lại. Thông số và kết quả thực tế cần được lấy từ các file được sinh ra, không điền số liệu giả vào báo cáo.
