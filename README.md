# FashionMNIST Lab 1 — Nhóm 4

Pipeline PyTorch phân loại ảnh thời trang thành 10 lớp bằng MLP.
Đã hoàn thành huấn luyện A/B/C/D, chọn bằng validation, đánh giá
test và đóng gói checkpoint cuối.

## Kết quả

- Cấu hình chọn: D, hidden_units=256, learning_rate=0.001.
- Best epoch: 9.
- Validation accuracy: 88,8333%.
- Test accuracy: 88,11% trên 10.000 ảnh.
- Test loss: 0.32580671586990356.

[Báo cáo đầy đủ](reports/Lab1_Report.md).

## Cài đặt và kiểm tra

Mở terminal tại thư mục gốc repository:

```bash
python -m venv .venv
```

Kích hoạt trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cài dependencies và kiểm tra:

```bash
python -m pip install -r requirements.txt
python main.py --mode doctor
python main.py --mode check
python -m unittest discover -s tests -v
```

Phiên bản môi trường đã chạy được ghi trong
[phần tích hợp](reports/sections/06_integration.md).
requirements.txt chưa khóa phiên bản. Chưa kiểm chứng cài đặt
trong môi trường Python mới.

## Kết quả đã đóng gói

- outputs/experiment_results.csv: so sánh A/B/C/D.
- outputs/runs/: config, metadata, history và loss curves.
- outputs/final/best_model.pth: checkpoint cuối.
- outputs/final/: selection, test metrics, CSV dự đoán, ảnh,
  reload check và provenance.
- outputs/smoke_report.json: kiểm tra tích hợp một batch.

Checkpoint riêng của từng run không được đóng gói.
Các đường dẫn tuyệt đối trong artifact ghi lại máy chạy gốc;
khi sử dụng, truyền đường dẫn checkpoint trong checkout hiện tại.

## Nạp checkpoint cuối

```python
from src.predict import load_model

model, metadata = load_model(
    "outputs/final/best_model.pth", "cpu"
)
```

SHA-256 được ghi trong outputs/final/selection_provenance.json.

## Chạy lại thí nghiệm

Các mode có sẵn: prepare, train, select, evaluate, smoke, plot.
Xem main.py --help và docs/TEAM_CONTRACT.md.

Luôn chạy tại thư mục gốc. Khi huấn luyện lại, dùng bản sao config
với output_root mới để giữ nguyên kết quả đã đóng gói.
Không chạy select trên thư mục final đã chứa checkpoint.
Mode evaluate ghi output cạnh checkpoint được truyền vào,
nên dùng bản sao checkpoint nếu muốn giữ nguyên artifact gốc.

Không dùng test để chọn cấu hình hoặc epoch.

## Kiểm chứng

Checkout sạch tại commit 01f734b đã qua mode check, 13 unittest,
đối chiếu SHA-256, metadata và forward checkpoint.
Smoke và reload đã PASS trên máy thực hiện.
Chi tiết và giới hạn nằm trong báo cáo TV6.
