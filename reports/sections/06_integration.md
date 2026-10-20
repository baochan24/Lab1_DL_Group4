# Tích hợp, lưu–tải và tái lập — TV6

## Thiết kế đã triển khai

Main điều phối module qua giao diện chung; không tải data hoặc train khi import. Cấu hình lưu cùng mỗi run; seed/device được chọn trước khi tạo mô hình. Split được kiểm trước train để phát hiện giao nhau, trùng chỉ số và sai số mẫu.

Metadata phân biệt running, failed, trained và completed; trạng thái vẽ riêng giúp phục hồi hình mà không chạy lại training. JSON ghi atomically, từ chối số NaN/Inf. Chọn checkpoint cuối kiểm nguồn run và ghi SHA256.

Load_model tạo lại MLP từ model_config, nạp state_dict với map_location và weights_only=True, sau đó eval. Schema kiểm preprocessing/data_root, tên lớp, epoch và val_metrics. Check_reload so logits/nhãn trên cùng batch và cùng trọng số nguồn. Smoke integration chỉ kiểm ghép một batch, không đánh giá chất lượng test.

## Kiểm chứng đã thực hiện trong lúc chuẩn bị

13 kiểm tra độc lập bằng unittest đã qua: schema checkpoint, JSON, split và vòng đời train giả lập qua giao diện thành viên trong thư mục tạm. Các fixture không phải kết quả FashionMNIST. Kiểm cú pháp và mode check đã qua. Doctor tại môi trường trợ lý báo thiếu torch/torchvision, nên chưa có bằng chứng train hoặc model reload thực tế.

## Cần bổ sung khi nhóm bàn giao

TODO TV6: Python/PyTorch/Torchvision và phần cứng máy train; lệnh cài; kết quả doctor/smoke; nguồn checkpoint cuối/run/epoch; kiểm chứng reload thật; cách lấy checkpoint; một lần chạy lại từ clone sạch. Chưa điền accuracy hoặc tuyên bố hoàn tất khi chưa có dữ liệu thực nghiệm.

## Tài liệu tham khảo

https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
