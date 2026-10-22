"""TV6 — tiện ích nền; không import torch ở cấp module."""
from pathlib import Path
import json
import random
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[1]

def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".json-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config phải là JSON object")
    required = {"run_id", "data_root", "split_indices_path", "train_size", "val_size",
                "seed", "batch_size", "num_workers", "transform", "model_name",
                "hidden_units", "optimizer", "learning_rate", "epochs", "device", "output_root"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Thiếu config: {sorted(missing)}")
    run_id = config["run_id"]
    if not isinstance(run_id, str) or not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in run_id):
        raise ValueError("run_id chỉ dùng chữ, số, dấu _ hoặc -")
    for key in ("train_size", "val_size", "batch_size", "hidden_units", "epochs"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} phải là số nguyên dương")
    for key in ("seed", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} phải là số nguyên không âm")
    import math
    lr = config["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate phải là số dương hữu hạn")
    if config["train_size"] + config["val_size"] != 60000:
        raise ValueError("train_size + val_size phải bằng 60000")
    for key in ("data_root", "split_indices_path", "output_root"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} phải là đường dẫn không rỗng")
    for key, expected in (("transform", "to_tensor"), ("model_name", "mlp"), ("optimizer", "adam")):
        if config[key] != expected:
            raise ValueError(f"Contract nền yêu cầu {key}={expected}")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device phải là auto, cpu hoặc cuda")
    return config

def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Không cam kết bitwise-identical giữa mọi phần cứng/phiên bản.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def resolve_device(name="auto"):
    import torch
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("Device phải là auto, cpu hoặc cuda")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("Đã chọn CUDA nhưng môi trường chưa có CUDA khả dụng")
    return torch.device(name)

def collect_environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "matplotlib", "pandas"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "commit_sha": commit}


def validate_split(split, config):
    """Kiểm tra file bàn giao TV1 trước khi train, không tự sửa data của TV1."""
    if not isinstance(split, dict) or split.get("dataset_size") != 60000:
        raise ValueError("Split phải có dataset_size=60000")
    if split.get("seed") != config["seed"]:
        raise ValueError("Seed split không khớp config")
    seen = set()
    for key, expected in (("train_indices", config["train_size"]), ("val_indices", config["val_size"])):
        indices = split.get(key)
        if not isinstance(indices, list) or len(indices) != expected:
            raise ValueError(f"{key} không đủ số mẫu theo config")
        if any(type(i) is not int or not 0 <= i < 60000 for i in indices):
            raise ValueError(f"{key} chứa chỉ số sai miền/kiểu")
        values = set(indices)
        if len(values) != len(indices) or seen.intersection(values):
            raise ValueError("Split có chỉ số lặp hoặc train/validation giao nhau")
        seen.update(values)
    if len(seen) != 60000:
        raise ValueError("Split chưa phủ toàn bộ train gốc")
    return split


def diagnose_environment():
    """Import và tính tensor thật; không chỉ kiểm package đã cài."""
    import importlib
    result = collect_environment()
    checks = {}
    modules = {}
    for name in ("numpy", "matplotlib", "pandas", "torch", "torchvision"):
        try:
            modules[name] = importlib.import_module(name)
            checks[name] = {"ok": True, "version": str(getattr(modules[name], "__version__", "unknown"))}
        except Exception as exc:
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if "torch" in modules:
        try:
            torch = modules["torch"]
            x = torch.ones(2, 3)
            if float(x.sum()) != 6.0:
                raise RuntimeError("Tensor check sai kết quả")
            result["cuda_available"] = bool(torch.cuda.is_available())
            result["gpu_name"] = torch.cuda.get_device_name(0) if result["cuda_available"] else None
            checks["tensor_operation"] = {"ok": True}
        except Exception as exc:
            checks["tensor_operation"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result.update({"checks": checks, "ready": all(x["ok"] for x in checks.values()),
                   "scope": "Environment only; no dataset or training verification"})
    return result
