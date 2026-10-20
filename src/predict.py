"""TV6 — nền lưu/tải; kiểm tra thực tế sau khi TV2/TV3 hoàn thành."""
from pathlib import Path
from .utils import project_path

def load_model(checkpoint_path, device):
    import torch
    from .model import MLP
    path = project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có checkpoint: {path}")
    # Checkpoint của nhóm chỉ gồm state_dict và metadata kiểu cơ bản.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    validate_checkpoint_metadata(checkpoint)
    model_config = checkpoint["model_config"]
    model = MLP(hidden_units=model_config["hidden_units"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return model, metadata

def predict_batch(model, images, device):
    import torch
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28) or images.shape[0] == 0:
        raise ValueError("Ảnh cần tensor không rỗng [B,1,28,28]")
    if images.dtype != torch.float32 or not torch.isfinite(images).all():
        raise ValueError("Ảnh phải float32 và hữu hạn")
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        if tuple(logits.shape) != (images.shape[0], 10) or not torch.isfinite(logits).all():
            raise ValueError("Model cần trả logits hữu hạn [B,10]")
        return {"logits": logits.cpu(), "labels": logits.argmax(dim=1).cpu()}


def check_reload(model, images, checkpoint_path, device, output_path):
    """TV6: so model hiện tại với đúng checkpoint của nó, không dùng model last epoch.
    Gọi ngay sau khi lưu best hoặc khi cả hai model cùng nạp đúng best checkpoint.
    Lần đầu kiểm tra nên làm trong smoke integration cùng một model trước/sau save.
    """
    import torch
    from .utils import save_json
    before = predict_batch(model, images, device)
    loaded, metadata = load_model(checkpoint_path, device)
    after = predict_batch(loaded, images, device)
    logits_match = torch.allclose(before["logits"], after["logits"], rtol=1e-5, atol=1e-6)
    labels_match = torch.equal(before["labels"], after["labels"])
    result = {"logits_match": bool(logits_match), "labels_match": bool(labels_match),
              "rtol": 1e-5, "atol": 1e-6, "num_samples": int(images.shape[0]),
              "run_id": metadata["run_id"], "epoch": metadata["epoch"],
              "checkpoint_path": str(project_path(checkpoint_path)),
              "passed": bool(logits_match and labels_match)}
    save_json(output_path, result)
    if not result["passed"]:
        raise ValueError("Reload check thất bại; xem JSON và kiểm tra model/checkpoint nguồn")
    return result


def validate_checkpoint_metadata(checkpoint):
    """Kiểm schema không cần torch; không chứng minh state_dict load được."""
    import math
    from collections.abc import Mapping
    required = {"model_state_dict", "model_config", "preprocessing", "class_names", "run_id", "epoch", "val_metrics"}
    if not isinstance(checkpoint, dict) or required - checkpoint.keys():
        raise ValueError("Checkpoint thiếu khóa theo TEAM_CONTRACT")
    if not isinstance(checkpoint["model_state_dict"], Mapping) or not checkpoint["model_state_dict"]:
        raise ValueError("model_state_dict phải là mapping có trọng số")
    cfg = checkpoint["model_config"]
    if not isinstance(cfg, dict) or cfg.get("model_name") != "mlp" or type(cfg.get("hidden_units")) is not int or cfg["hidden_units"] <= 0:
        raise ValueError("model_config phải là mlp với hidden_units nguyên dương")
    preprocessing = checkpoint["preprocessing"]
    if not isinstance(preprocessing, dict) or preprocessing.get("transform") != "to_tensor":
        raise ValueError("Checkpoint chưa có transform to_tensor")
    if not isinstance(preprocessing.get("data_root"), str) or not preprocessing["data_root"]:
        raise ValueError("preprocessing thiếu data_root để tái tạo data")
    names = checkpoint["class_names"]
    if not isinstance(names, list) or len(names) != 10 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 10:
        raise ValueError("Checkpoint cần 10 tên lớp khác nhau theo thứ tự nhãn")
    if not isinstance(checkpoint["run_id"], str) or not checkpoint["run_id"]:
        raise ValueError("Checkpoint thiếu run_id")
    if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 1:
        raise ValueError("Epoch phải là số nguyên từ 1")
    metrics = checkpoint["val_metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("val_metrics phải là dict")
    for key in ("loss", "accuracy"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"val_metrics.{key} phải hữu hạn, không âm")
    if metrics["accuracy"] > 1:
        raise ValueError("Accuracy trong checkpoint phải ở thang 0–1")
    return checkpoint
