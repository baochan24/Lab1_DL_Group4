# Quy tắc thực hiện — Lab 1 — FashionMNIST

## Quy ước dữ liệu và đánh giá
1. Chỉ dùng tập **train** để tối ưu trọng số; **validation** để chọn cấu hình/epoch; **test** chỉ để báo cáo cuối.
2. Cố định seed 42 theo các file cấu hình và không thay đổi split giữa những run cần so sánh.
3. Không coi hình PCA hai chiều là bằng chứng về độ chính xác phân loại; PCA chỉ dùng để khám phá cấu trúc dữ liệu.
4. Các kết quả trong báo cáo phải dẫn tới file gốc (`outputs/`), không tự tạo hoặc làm tròn để thay đổi thứ hạng.
5. Không ghi đè checkpoint, `history.csv`, `metrics.json` hoặc cấu hình đã lưu khi chỉ thực hiện bổ sung tài liệu.
6. Các nhận xét về cải thiện phải nêu điều kiện thí nghiệm: thời gian chạy, seed, kiến trúc, batch size và số epoch.
7. Lưu ảnh trình bày tại `reports/figures/`; ảnh khám phá mới tại `outputs/eda/`.
8. Không đưa file dữ liệu lớn, token, secrets hoặc môi trường `.venv/` lên GitHub.

## Quy tắc trình bày
- Chú thích hình/bảng và ghi rõ nguồn số liệu.
- Phân biệt kết quả đã chạy với bước đang chuẩn bị.
- Mỗi kết luận phải tương ứng với một thí nghiệm thực tế.
- Phân loại rõ nhãn `best validation accuracy` và `best validation loss`, không dùng thay thế tùy ý.
- Mọi thay đổi mã nguồn lõi cần kiểm tra lại; phần bổ sung này không yêu cầu train lại.
