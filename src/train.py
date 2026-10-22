"""TV3 - training loop, validation, history, checkpoint and metadata."""

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.evaluate import evaluate
from src.utils import project_path


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _read_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata phải là JSON object: {metadata_path}")
    return value


def _write_metadata(metadata_path, metadata):
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(metadata_path)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch and return sample-weighted loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader rỗng; không thể train một epoch.")

    return {
        "loss": float(total_loss / total_samples),
        "accuracy": float(total_correct / total_samples),
    }


def fit(model, train_loader, val_loader, config, run_dir):
    """Train the model, log metrics and save the best validation checkpoint.

    Best epoch: maximum validation accuracy; break ties by lower validation loss;
    if both values tie, keep the earlier epoch. Does not use the test loader.
    """
    run_dir = project_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best_model.pth"
    metadata_path = run_dir / "metadata.json"

    required = {
        "epochs", "learning_rate", "optimizer", "hidden_units", "data_root",
        "run_id", "class_names", "transform",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Config TV3 còn thiếu khóa: {missing}")
    if history_path.exists() or checkpoint_path.exists():
        raise FileExistsError(
            f"Đã có history/checkpoint trong {run_dir}; không ghi đè run cũ. "
            "Hãy dùng run_id mới."
        )

    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("config['epochs'] phải là số nguyên dương.")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("config['learning_rate'] phải là số dương hữu hạn.")
    if str(config["optimizer"]).lower() != "adam":
        raise ValueError("Hợp đồng TV3 hiện yêu cầu optimizer='adam'.")
    if config["transform"] != "to_tensor":
        raise ValueError("Hợp đồng TV3 hiện yêu cầu transform='to_tensor'.")
    if len(config["class_names"]) != 10:
        raise ValueError("FashionMNIST cần đúng 10 class_names.")

    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Model không có parameter để train.") from exc

    # These are constructed exactly once per run, not once per epoch.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=float(learning_rate))

    started_at = _now_iso()
    start_clock = time.perf_counter()
    metadata = _read_metadata(metadata_path)
    metadata.update({
        "training_status": "running",
        "training_started_at": started_at,
        "training_ended_at": None,
        "training_duration_seconds": None,
        "best_epoch": None,
        "best_val_loss": None,
        "best_val_accuracy": None,
        "training_error": None,
    })
    _write_metadata(metadata_path, metadata)

    best_epoch = None
    best_accuracy = float("-inf")
    best_loss = float("inf")
    columns = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]

    try:
        with history_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            csv_file.flush()

            for epoch in range(1, epochs + 1):
                train_metrics = train_one_epoch(
                    model, train_loader, loss_fn, optimizer, device
                )
                val_metrics = evaluate(model, val_loader, loss_fn, device)
                train_loss = float(train_metrics["loss"])
                train_accuracy = float(train_metrics["accuracy"])
                val_loss = float(val_metrics["loss"])
                val_accuracy = float(val_metrics["accuracy"])

                values = (train_loss, train_accuracy, val_loss, val_accuracy)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Metric NaN/Inf tại epoch {epoch}.")
                if not (0.0 <= train_accuracy <= 1.0 and 0.0 <= val_accuracy <= 1.0):
                    raise ValueError("Accuracy phải nằm trong [0, 1].")

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
                writer.writerow(row)
                csv_file.flush()

                print(
                    f"Epoch [{epoch}/{epochs}] "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_accuracy:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_acc={val_accuracy:.4f}"
                )

                is_better = (
                    best_epoch is None
                    or val_accuracy > best_accuracy
                    or (val_accuracy == best_accuracy and val_loss < best_loss)
                )
                if is_better:
                    best_epoch = int(epoch)
                    best_accuracy = val_accuracy
                    best_loss = val_loss
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "model_name": "mlp",
                            "hidden_units": int(config["hidden_units"]),
                        },
                        "preprocessing": {
                            "transform": "to_tensor",
                            "data_root": str(config["data_root"]),
                        },
                        "class_names": list(config["class_names"]),
                        "run_id": str(config["run_id"]),
                        "epoch": int(epoch),
                        "val_metrics": {
                            "loss": float(val_loss),
                            "accuracy": float(val_accuracy),
                        },
                    }
                    temp_checkpoint = checkpoint_path.with_suffix(".pth.tmp")
                    torch.save(checkpoint, temp_checkpoint)
                    temp_checkpoint.replace(checkpoint_path)

        if best_epoch is None or not checkpoint_path.is_file():
            raise RuntimeError("Không tạo được best checkpoint trong quá trình train.")

        ended_at = _now_iso()
        duration = float(time.perf_counter() - start_clock)
        metadata = _read_metadata(metadata_path)
        metadata.update({
            "training_status": "trained",
            "training_started_at": started_at,
            "training_ended_at": ended_at,
            "training_duration_seconds": duration,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_loss),
            "best_val_accuracy": float(best_accuracy),
            "training_error": None,
        })
        _write_metadata(metadata_path, metadata)

        return {
            "history_path": str(history_path.resolve()),
            "best_checkpoint_path": str(checkpoint_path.resolve()),
            "best_epoch": int(best_epoch),
        }

    except Exception as exc:
        ended_at = _now_iso()
        duration = float(time.perf_counter() - start_clock)
        metadata = _read_metadata(metadata_path)
        metadata.update({
            "training_status": "failed",
            "training_started_at": started_at,
            "training_ended_at": ended_at,
            "training_duration_seconds": duration,
            "training_error": f"{type(exc).__name__}: {exc}",
        })
        _write_metadata(metadata_path, metadata)
        raise
