"""TV6 — kiểm tra ghép một batch sau khi TV1–TV4 bàn giao; không tạo kết quả lab."""
from pathlib import Path
import tempfile
from .utils import project_path, load_json, save_json, validate_split, set_seed, resolve_device


def smoke_integration(config, device_name="auto", report_path=None):
    import torch
    from .data import build_dataloaders
    from .model import MLP
    from .train import train_one_epoch
    from .evaluate import evaluate
    from .predict import check_reload
    split = project_path(config["split_indices_path"])
    if not split.is_file():
        raise FileNotFoundError("Cần TV1 chuẩn bị split trước smoke integration")
    validate_split(load_json(split), config)
    set_seed(config["seed"])
    device = resolve_device(device_name)
    train_loader, _val, _test, class_names = build_dataloaders(config)
    images, labels = next(iter(train_loader))
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28) or images.dtype != torch.float32 or not torch.isfinite(images).all():
        raise ValueError("Batch ảnh sai shape/dtype hoặc không hữu hạn")
    if images.shape[0] == 0 or tuple(labels.shape) != (images.shape[0],) or labels.dtype != torch.int64:
        raise ValueError("Batch nhãn sai shape/dtype hoặc batch rỗng")
    if labels.min() < 0 or labels.max() > 9 or images.min() < 0 or images.max() > 1:
        raise ValueError("Nhãn/pixel ngoài miền contract")
    model = MLP(hidden_units=config["hidden_units"]).to(device)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()
    # Dùng đúng hàm TV3 và TV4. Chỉ một batch train, không lấy metrics này làm test.
    train_metrics = train_one_epoch(model, [(images, labels)], loss_fn, optimizer, device)
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    if not any(not torch.equal(a, b) for a, b in zip(before, after)):
        raise ValueError("Không thấy trọng số thay đổi sau optimizer step")
    eval_before = [p.detach().clone() for p in model.parameters()]
    metrics = evaluate(model, [(images, labels)], loss_fn, device)
    if any(not torch.equal(a, b.detach()) for a, b in zip(eval_before, model.parameters())):
        raise ValueError("evaluate đã thay đổi trọng số")
    import math
    for metric_set in (train_metrics, metrics):
        if not isinstance(metric_set, dict):
            raise ValueError("Metrics phải là dict")
        for key in ("loss", "accuracy"):
            value = metric_set.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Metrics {key} không hợp lệ")
        if metric_set["accuracy"] > 1:
            raise ValueError("Accuracy phải trong 0–1")
    with tempfile.TemporaryDirectory(prefix="lab1-smoke-") as temp:
        checkpoint = Path(temp) / "smoke.pth"
        torch.save({"model_state_dict": model.state_dict(),
                    "model_config": {"model_name": "mlp", "hidden_units": config["hidden_units"]},
                    "preprocessing": {"transform": "to_tensor", "data_root": config["data_root"]},
                    "class_names": list(class_names), "run_id": "smoke", "epoch": 1,
                    "val_metrics": metrics}, checkpoint)
        reload_result = check_reload(model, images, checkpoint, device, Path(temp) / "reload.json")
    # Không giữ đường dẫn checkpoint tạm đã bị xóa trong report.
    reload_result.pop("checkpoint_path", None)
    result = {"purpose": "one-batch integration only; NOT validation/test performance", "passed": True,
              "batch_shape": list(images.shape), "device": str(device), "weights_changed": True,
              "evaluate_preserved_weights": True, "reload": reload_result}
    if report_path:
        save_json(project_path(report_path), result)
    return result
