# data — TV1

# 01. Báo cáo Dữ liệu và Tiền xử lý (Data & Preprocessing)

## 1. Mục tiêu
* Triển khai mô-đun hóa quy trình xử lý dữ liệu từ notebook ban đầu vào file `src/data.py`.
* Xây dựng bộ chỉ số phân chia (split indices) cố định $54.000$ mẫu Train và $6.000$ mẫu Validation từ $60.000$ ảnh gốc, lưu vết để dùng chung cho mọi cấu hình thử nghiệm nhằm ngăn rò rỉ dữ liệu.
* Khởi tạo giao diện `build_dataloaders(config)` chuẩn hóa để cung cấp dữ liệu cho các thành viên TV2, TV3, TV4, TV5, TV6.
* Xuất đầy đủ các file báo cáo thống kê, hình ảnh trực quan hóa và kiểm tra điều kiện dữ liệu theo yêu cầu.

---

## 2. Cách làm (Quy trình thực hiện)
1. **Nạp dữ liệu & Transforms**: Sử dụng `torchvision.datasets.FashionMNIST` nạp 60.000 ảnh gốc và 10.000 ảnh Test. Áp dụng `transforms.ToTensor()` giữ nguyên kích thước $28 \times 28$ pixels.
2. **Cố định Split Index**: Viết hàm `get_or_create_split_indices()` sử dụng `torch.Generator().manual_seed(42)` để tạo danh sách hoán vị ngẫu nhiên. Tách $54.000$ chỉ số cho tập Train và $6.000$ chỉ số cho tập Validation, lưu vào `outputs/split_indices.json`.
3. **Đóng gói DataLoader**: Thiết lập `train_loader` có `shuffle=True` (kèm `generator` cố định seed), `val_loader` và `test_loader` có `shuffle=False`.
4. **Trích xuất thông số & Thống kê**:
   * Kiểm tra thông số batch đầu tiên (shape, dtype, pixel range).
   * Đếm số lượng ảnh của 10 lớp trên từng tập (Train, Val, Test) và xuất ra file CSV.
   * Vẽ lưới 16 ảnh xám kèm nhãn tiếng Anh thực tế để kiểm tra hiển thị.

---

## 3. Output thực tế

| Tên File | Loại File | Nội dung thực tế |
|---|---|---|
| `src/data.py` | Python Script | Mã nguồn triển khai hai hàm `prepare_data(config)` và `build_dataloaders(config)`. |
| `outputs/split_indices.json` | JSON | Lưu `seed: 42`, `dataset_size: 60000`, `train_indices` (54.000 phần tử), `val_indices` (6.000 phần tử). |
| `outputs/data_summary.json` | JSON | `train_samples`: 54000, `val_samples`: 6000, `test_samples`: 10000; `image_shape`: [1, 28, 28]; `batch_image_shape`: [64, 1, 28, 28]; `pixel_min`: 0.0, `pixel_max`: 1.0; `image_dtype`: torch.float32; `label_dtype`: torch.int64; `split_is_disjoint`: true. |
| `outputs/class_distribution.csv` | CSV | Bảng thống kê 10 lớp: Train (~5.400 mẫu/lớp), Val (~600 mẫu/lớp), Test (1.000 mẫu/lớp). |
| `outputs/sample_images.png` | PNG | Lưới $4 \times 4$ hiển thị 16 ảnh mẫu xám kèm tiêu đề dạng `ID: Class_Name`. |

---

## 4. Cách kiểm tra (Validation & Sanity Check)
* **Kiểm tra tính không giao nhau của Split**: Xác minh tập hợp chỉ số `set(train_indices) & set(val_indices)` có kết quả rỗng và tổng số phần tử bằng đúng $60.000$ mẫu gốc.
* **Kiểm tra thông số Tensor Batch**:
  * Chạy `next(iter(train_loader))` xác nhận shape ảnh là `[64, 1, 28, 28]` và nhãn là `[64]`.
  * Giá trị pixel nhỏ nhất $\ge 0.0$ và lớn nhất $\le 1.0$, kiểu dữ liệu ảnh là `torch.float32`, kiểu dữ liệu nhãn là `torch.int64`.
* **Tái tạo chỉ số**: Chạy lại hàm `prepare_data` khi file `split_indices.json` đã tồn tại để đảm bảo chương trình nạp lại chỉ số cũ chứ không ghi đè split mới.

---

## 5. Nhận xét & Giải thích Kỹ thuật

### 5.1 `Dataset` khác `DataLoader` thế nào?
* **`Dataset`**: Lưu trữ dữ liệu thô và các chuyển đổi (transforms), cung cấp cơ chế truy cập từng mẫu dữ liệu đơn lẻ `(image, label)` thông qua chỉ số `dataset[i]`.
* **`DataLoader`**: Lớp quản lý đóng gói `Dataset` thành các lô (batches), thực hiện xáo trộn (`shuffle`), và hỗ trợ tải dữ liệu song song đa tiến trình (`num_workers`).

### 5.2 Ý nghĩa kích thước `[B, 1, 28, 28]`
* **`B` (Batch size)**: Số lượng ảnh đưa vào mô hình xử lý cùng một lúc (mặc định là 64).
* **`1` (Channels)**: Số kênh màu của ảnh (ảnh xám Grayscale chỉ có 1 kênh).
* **`28` (Height)** & **`28` (Width)**: Chiều cao và chiều rộng của ảnh tính theo pixel.

### 5.3 Vì sao Train cần `shuffle=True` còn Validation/Test cần `shuffle=False`?
* **Tập Train (`shuffle=True`)**: Đảo thứ tự mẫu sau mỗi epoch giúp mô hình không bị học thuộc chuỗi xuất hiện của dữ liệu, giảm thiểu overfitting và giúp thuật toán Gradient Descent tìm điểm tối ưu tốt hơn.
* **Tập Validation/Test (`shuffle=False`)**: Cần giữ nguyên thứ tự cố định để kết quả đánh giá nhất quán qua các lượt chạy, giúp so sánh chính xác giữa các mô hình và thuận tiện khi đối chiếu ảnh bị dự đoán sai.

### 5.4 Vai trò của `ToTensor()`
* Chuyển cấu trúc ảnh PIL Image hoặc mảng NumPy `(H x W x C)` sang `torch.FloatTensor` `(C x H x W)`.
* Tự động chuẩn hóa (scale) giá trị pixel từ khoảng $[0, 255]$ về khoảng số thực $[0.0, 1.0]$ bằng cách chia cho $255.0$.

### 5.5 Lý do chưa cần làm sạch sâu (Data Cleaning)
Dữ liệu FashionMNIST gốc đã được Zalando Research chuẩn hóa sẵn (kích thước $28 \times 28$ đồng nhất, vật thể được căn giữa, nền đen chữ trắng, không xuất hiện giá trị bị khuyết/null). Do đó, việc can thiệp lọc ảnh hay xóa dữ liệu thủ công là không cần thiết và có thể làm mất tính khách quan của tập benchmark.
