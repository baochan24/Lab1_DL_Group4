# TV6 kiêm leader — việc đã chuẩn bị và cách sử dụng

## 1. Bắt đầu trên branch riêng

Trong repo của em, commit/lưu công việc đang làm trước rồi fetch. Nếu dùng branch cập nhật TV6 được cung cấp qua draft PR thì checkout đúng branch đó; nếu áp dụng thủ công, chỉ lấy các file TV6 đã review vào feature/integration. Không chép đè nguyên project lên code các bạn vừa bàn giao.

```bash
git status
git fetch origin
```

Không tự chạy git init trên repo đã clone. Không push thẳng main.

## 2. Các việc em làm được ngay

| Việc | Lệnh / nơi làm | Điều kiện đạt |
|---|---|---|
| Kiểm bộ khung | python main.py --mode check | Không thiếu file/giao diện; TODO còn lại hiển thị rõ |
| Kiểm phần TV6 độc lập | python -m unittest discover -s tests -v | Các test qua; chưa chứng minh ML |
| Kiểm môi trường | python main.py --mode doctor --report outputs/diagnostics/environment.json | ready=true sau import và tensor operation thật |
| Chuẩn bị nghiệm thu | Điền bảng tiến độ ở cuối file này | Mỗi phần có người, PR và output kiểm chứng |
| Chuẩn bị báo cáo | reports/sections/06_integration.md | Phần mô tả đúng code; chưa ghi kết quả chưa chạy |

Lệnh doctor dùng được trước khi TV2 hoàn thiện MLP. Cài torch/torchvision theo https://pytorch.org/get-started/locally/ rồi cài requirements.txt. CPU đủ cho mốc ghép; CUDA chỉ chốt sau khi thấy phần cứng/driver của máy train. Chưa thể khóa phiên bản máy em từ môi trường của trợ lý.

Windows: `python -m venv .venv`, kích hoạt `.\.venv\Scripts\Activate.ps1`, chọn interpreter tương ứng trong VS Code. Nếu PowerShell chặn kích hoạt, có thể chạy trực tiếp `.\.venv\Scripts\python.exe` thay cho python; không bắt buộc đổi execution policy.

Sau doctor thành công trên môi trường sạch: `python -m pip freeze > requirements-lock.txt`. Ghi Python version, CPU/GPU và lệnh cài PyTorch trong README. Lockfile chỉ tạo từ môi trường thật đã chạy, không điền phiên bản dự đoán.

## 3. Trạng thái bàn giao quan sát tại main

TV1 đã bàn giao data.py trong commit 1c9fb952c9020507ff2d41c7aab089a9ce54772b. Có thêm pandas; requirements đã bổ sung. Các phần model/train/evaluate/experiments/visualize còn TODO trong snapshot dùng chuẩn bị TV6. Không kết luận TV1 đã đạt mọi tiêu chí chỉ vì có PR merge.

**Cần review với TV1:** kiểm tra cũ chỉ dùng tổng số split <=60000 chưa đủ phát hiện trùng nội bộ/thiếu mẫu/sai seed. TV6 đã thêm kiểm tra độc lập trước train; TV1 nên bổ sung cùng tiêu chí khi prepare. Code data.py đang dùng đường dẫn tương đối CWD; luôn chạy các lệnh từ thư mục gốc repo như README, hoặc trao đổi TV1 dùng project_path để hỗ trợ notebook/chạy từ chỗ khác. TV6 không sửa thay data.py trong cập nhật này.

## 4. Ghép sớm khi có TV1–TV4

```bash
python main.py --mode prepare --config configs/A.json
python main.py --mode smoke --config configs/A.json --device cpu --report outputs/diagnostics/smoke.json
```

Smoke lấy một batch train, gọi train_one_epoch của TV3, kiểm trọng số thay đổi, gọi evaluate của TV4 và kiểm trọng số không thay đổi, lưu checkpoint tạm rồi nạp bằng predict.py để so logits/nhãn. Checkpoint tạm bị xóa. Report ghi rõ mục đích kiểm ghép, không tính là validation/test accuracy.

Nếu thiếu module, thư viện hoặc hàm còn NotImplementedError: ghi rõ phần đang chờ; không sửa output để pass. Smoke chưa chạy thành công trong lần chuẩn bị này vì môi trường trợ lý chưa có torch và thành viên khác còn TODO.

## 5. Khi chạy baseline thật

```bash
python main.py --mode train --config configs/A.json
```

Main từ chối run đã có, ghi running/failed/trained/completed trong metadata. Nếu lỗi train: giữ output lỗi để xem, đổi run_id khi chạy lại. Nếu train xong mà vẽ lỗi: không train lại A; sau khi TV4 sửa hàm, chạy:

```bash
python main.py --mode plot --history outputs/runs/A/history.csv
```

Mode plot tạo ảnh; không tự đổi metadata từ trained thành completed vì leader còn phải kiểm hình. Ghi nhận kết quả kiểm trong PR/biên bản tích hợp.

## 6. Khi TV5 bàn giao đủ run

```bash
python main.py --mode select --config configs/A.json --runs A B C D
python main.py --mode evaluate --checkpoint outputs/final/best_model.pth --device cpu
```

Select dùng hàm TV5, chép đúng best_model.pth và ghi SHA256. Không ghi đè checkpoint cuối đã có; nếu chọn lại phải lưu bản cũ có chủ đích và xem các hình/test cũ có còn tương ứng không. Evaluate vẫn cần TV4 triển khai; không tune theo test.

Kiểm tra reload của mô hình thật cần so cùng model/epoch/weights, eval và cùng input. Không so last epoch với best checkpoint khác epoch. Bằng chứng smoke không thay thế việc xác nhận checkpoint cuối dùng trong báo cáo.

## 7. Quản lý raw data đã được commit

Repo có raw files trong notebooks/data. .gitignore chỉ ngăn file chưa được theo dõi, không tự bỏ file đã tracked. Sau khi đã kiểm tra đây là dữ liệu tải lại được và thống nhất với nhóm, leader có thể tự thực hiện một commit riêng:

```bash
git ls-files notebooks/data
git rm -r --cached -- notebooks/data
git status
```

`--cached` bỏ khỏi index, giữ file local; chỉ commit sau khi review staged changes. Không xóa toàn bộ outputs hoặc rewrite lịch sử đang dùng chung. Phần dọn này chưa được thực hiện tự động trong cập nhật TV6.

## 8. Bảng kiểm leader

| Phần | Người / PR | Đã chạy gì? | Còn chờ |
|---|---|---|---|
| Data/split | TV1 / PR #1 đã merge | Chờ kiểm trên máy tích hợp | Split validation và đường dẫn |
| Forward | TV2 / … | … | … |
| Train một batch | TV3 / … | … | … |
| Evaluate/vẽ | TV4 / … | … | … |
| So sánh run | TV5 / … | … | … |
| Môi trường + reload | TV6 / … | Check nền và test độc lập đã có | Doctor/smoke trên máy train |

## 9. Checklist cuối em chịu trách nhiệm

- [ ] Code đã review và ghép bằng PR, giữ contract.
- [ ] Doctor thành công, môi trường được ghi lại và có lockfile thật.
- [ ] Smoke một batch thành công bằng code đã bàn giao.
- [ ] Run chính có config/history/checkpoint/metadata/hình thật.
- [ ] TV5 chọn theo validation; test khớp checkpoint nguồn.
- [ ] Checkpoint cuối tải lại và dự đoán được.
- [ ] Báo cáo khớp CSV, run_id, epoch và ảnh.
- [ ] Một bạn khác chạy theo README từ clone sạch được.

Chỉ đánh dấu sau khi kiểm chứng. Không tự viết lại core ML thay các thành viên để né chờ bàn giao.
