# HỢP ĐỒNG GHÉP CODE — phiên bản nền 1

Đây là quyết định kỹ thuật xuất phát của nhóm, không phải yêu cầu bổ sung của giảng viên. Mọi thay đổi giao diện phải cập nhật tài liệu và báo người sử dụng. Đọc docstring trong src để biết chữ ký hàm chính xác.

## Dữ liệu

FashionMNIST: 60.000 train gốc, 10.000 test; chia train gốc thành 54.000/6.000 bằng seed 42. TV1 lưu JSON với `seed`, `dataset_size=60000`, `train_indices`, `val_indices`; kiểm tra độ dài, chỉ số trong miền, không trùng, không giao và phủ đủ. Chia một lần, dùng lại mọi run. Tách tập random không bảo đảm số mẫu lớp bằng nhau tuyệt đối.

Ảnh float32 `[B,1,28,28]`, pixel 0–1 sau ToTensor; nhãn long `[B]`, 0–9. Batch cuối có thể nhỏ hơn 64. MLP trả logits `[B,10]`; không Softmax trước CrossEntropyLoss. Tên lớp lấy từ dataset.classes, giữ thứ tự. Chưa augmentation/resize/khử nhiễu. Không xóa mẫu khó tùy ý.

## Giao diện

| Chủ | Hàm | Return cố định |
|---|---|---|
| TV1 | prepare_data(config) | dict đường dẫn output thực tế |
| TV1 | build_dataloaders(config) | tuple 4 phần: train_loader, val_loader, test_loader, class_names |
| TV2 | MLP(hidden_units=128), forward(images) | logits tensor `[B,10]` |
| TV3 | train_one_epoch(model, loader, loss_fn, optimizer, device) | dict loss/accuracy Python float |
| TV3 | fit(model, train_loader, val_loader, config, run_dir) | dict history_path, best_checkpoint_path, best_epoch |
| TV4 | evaluate(model, loader, loss_fn, device) | dict loss/accuracy Python float |
| TV4 | collect_predictions(model, loader, device) | dict true_labels, predicted_labels, indices (list) |
| TV4 | evaluate_final(checkpoint_path, device) | dict đường dẫn output test/ảnh |
| TV4 | plot_history(history_csv, output_path) | str PNG path |
| TV4 | plot_predictions(model, dataset, indices, class_names, device, output_path) | str PNG path |
| TV5 | summarize_and_select(run_dirs, output_root) | dict results_path, selection_path, checkpoint_path |
| TV6 | load_model(checkpoint_path, device) | model ở eval, metadata dict |

Đường dẫn trả về là đường dẫn tuyệt đối dạng str để không lệch nơi gọi. Accuracy lưu 0–1, hiển thị thì nhân 100. Epoch bắt đầu 1.

## Phân định trách nhiệm

Main đặt seed, resolve device, tạo model mới và run_dir, ghi config/metadata. TV3 fit tạo loss và optimizer một lần, gọi evaluate của TV4, ghi history và checkpoint. Main của TV6 ghi thời gian/trạng thái vòng đời run và thời gian gọi fit; TV3 có thể bổ sung chi tiết epoch vào metadata nhưng không thay khóa của main. Không có optimizer thứ hai trong main. TV5 dùng lại pipeline, không viết train mới.

Model và tensor phải cùng device. model.train trước mỗi epoch; evaluate gọi eval/no_grad. Không import ngược train từ evaluate. Không train/tải khi import. Các hàm dùng ROOT/project_path từ utils; checkpoint data_root là đường dẫn tương đối project để mang sang máy khác.

## File output

- Run: outputs/runs/<run_id>/config.json, metadata.json, history.csv, best_model.pth, loss_curves.png.
- history.csv: epoch,train_loss,train_accuracy,val_loss,val_accuracy. Loss là trung bình theo số mẫu, không lấy trung bình các batch không cùng kích thước.
- Chọn epoch: val_accuracy cao nhất, hòa thì val_loss thấp hơn; nếu hòa cả hai giữ epoch trước.
- Chọn run: cùng quy tắc; hòa cả hai giữ run đầu trong danh sách run_dirs. Không chọn theo test.
- TV5 ghi experiment_results.csv và final/selection.json gồm run_id, best_epoch, checkpoint_path, val_metrics, criterion.
- Lệnh `main.py --mode select` gọi hàm TV5 rồi TV6 sao chép đúng checkpoint chọn vào outputs/final/best_model.pth (không train thêm). Không nhầm last epoch với best.
- TV4 test cuối và tạo final/test_metrics.json, test_predictions.csv, predictions.png, misclassified.png nếu có lỗi. Test predictions có test_index,true_label,predicted_label. Chỉ collect_predictions với loader không shuffle để chỉ số có ý nghĩa đúng thứ tự dataset.
- TV6 ghi final/reload_check.json bằng kiểm tra thực tế. Không tạo file này như bằng chứng trước khi chạy.
- Các file run A/B/C/D là kết quả sinh từ chương trình, không viết số giả.

## Checkpoint TV3 phải lưu

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_config": {"model_name": "mlp", "hidden_units": config["hidden_units"]},
    "preprocessing": {"transform": "to_tensor", "data_root": config["data_root"]},
    "class_names": config["class_names"],
    "run_id": config["run_id"],
    "epoch": epoch,
    "val_metrics": {"loss": float(val_loss), "accuracy": float(val_accuracy)}
}
```

Lưu bằng torch.save; dùng metadata kiểu cơ bản (str/int/float/list/dict), không nhét model object/numpy scalar vào metadata. load_model dùng weights_only=True, strict=True và map_location. Chốt cặp PyTorch/Torchvision hỗ trợ trên máy tham chiếu trước khi tích hợp train/reload.

Kiểm tra reload: cùng batch và model.eval, lấy logits trước khi lưu, nạp vào model mới, so torch.allclose với dung sai được ghi và nhãn phải giống. Ghi kết quả thật. Không chỉ kiểm tra file tồn tại.

## Tái lập và quy trình nhóm

Giữ cùng split, seed, transforms và epoch giữa các cấu hình; đặt lại seed và khởi tạo model/optimizer mới mỗi run. Seed không bảo đảm đồng nhất tuyệt đối giữa mọi phần cứng. Ghi package versions, device và commit. Khi so thời gian ghi rõ máy.

Chạy check nền trước khi PR. Ghép sớm một batch sau khi TV1–4 triển khai, không chờ cuối bài. Mode check chỉ kiểm tra nền/config/cú pháp, không chứng minh thuật toán đúng. Thiết lập branch/PR và review ít nhất một người; không thay code đang train. Không commit raw data, token, môi trường ảo hoặc checkpoint lớn. Checkpoint phải có kênh lưu bền vững và hướng dẫn lấy.

## Việc còn phải chốt trên máy thật

- [ ] Điền tên TV1–TV6 và người review từng PR.
- [ ] Chọn máy train/môi trường CPU hay CUDA.
- [ ] Cài cặp PyTorch/Torchvision, ghi version đã chạy; tạo lockfile.
- [ ] Chạy lại 5 bước dữ liệu và lưu split dùng chung.
- [ ] Chọn kênh lưu checkpoint và kết quả khi cloud kết thúc.
- [ ] Thống nhất deadline các mốc theo lịch nhóm.

## Nguồn đối chiếu

- Đề Practice 1, trang 19; workflow và tutorial PyTorch, trang 18 trong tài liệu giảng viên đã cung cấp.
- https://github.com/zalandoresearch/fashion-mnist
- https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- https://pytorch.org/get-started/locally/

## Bổ sung sau rà soát nền

- CLI có đủ check → prepare → train (A/B/C/D) → select → evaluate. Select không cần torch, nhưng cần outputs thật và hàm summarize_and_select của TV5 được hoàn thành.
- Main gọi plot_history của TV4 sau fit; khi triển khai train có thể kiểm tra riêng train_one_epoch trước khi phần vẽ sẵn sàng.
- Đã có check_reload(model, images, checkpoint_path, device, output_path) trong predict.py. TV6 phải sử dụng với đúng model nguồn và checkpoint tương ứng, không so model last epoch với checkpoint best của epoch khác.
- evaluate_final lấy batch_size/seed mặc định của baseline (64/42), ToTensor và data_root từ checkpoint. Với test không shuffle và không augmentation, các giá trị này không là tham số tuning. Cần thêm tham số mới thì cập nhật hợp đồng trước.
- Output root cho test cuối là thư mục cha của checkpoint cuối được truyền vào; các file test không hard-code sang một project/máy khác.
- TV4 evaluate_final gọi plot_predictions; không cần TV6 viết thêm một lệnh vẽ ảnh riêng. TV4 vẫn chịu trách nhiệm mọi ảnh và CSV test.
- summarize_and_select trả đường dẫn checkpoint nguồn có thật, không phải đường dẫn đích final chưa tạo; main sao chép sang final/best_model.pth.


## TV6 bổ sung trước tích hợp — 20/10/2026

Giữ nguyên mọi chữ ký TV1–TV5. Không phải thành viên khác đổi thuật toán để dùng phần bổ sung.

- Main kiểm split đúng số mẫu, seed, miền chỉ số, không lặp/giao và phủ đủ trước train. Nếu TV1 bàn giao split không đạt thì báo lỗi, không tự sửa split.
- `save_json` ghi atomically và từ chối NaN/Inf, tránh làm hỏng JSON cũ khi lỗi.
- Checkpoint phải chứa preprocessing.data_root, 10 tên lớp khác nhau, epoch nguyên dương và val_metrics hữu hạn/accuracy 0–1. TV3 vốn đã được yêu cầu các khóa này.
- Main ghi `running` trước fit, `failed` nếu fit lỗi, `trained` khi fit xong và `completed` sau vẽ. Nếu vẽ lỗi: status vẫn `trained`, plot_status=`failed`; dùng mode plot để phục hồi hình. duration_seconds của main là thời gian gọi fit, không so trực tiếp giữa phần cứng khác nhau.
- Select chỉ chấp nhận checkpoint nguồn là best_model.pth thuộc các run được so sánh, chép không ghi đè và ghi selection_provenance.json có SHA256. TV5 vẫn chịu trách nhiệm so sánh/chọn trên validation.
- src/integration.py thuộc TV6; chế độ smoke là kiểm tra ghép một batch, không thay training loop TV3 hoặc số liệu test TV4.
- Doctor/report chỉ là kiểm môi trường, không là bằng chứng hoàn thành lab.
