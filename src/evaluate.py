"""TV4 — metrics và test cuối; không import train.py."""

def evaluate(model, loader, loss_fn, device):
    """Trả loss trung bình theo số mẫu và accuracy trong [0, 1]."""
    import math
    import torch

    if getattr(loss_fn, "reduction", "mean") != "mean":
        raise ValueError("loss_fn phải dùng reduction='mean'")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            if not torch.isfinite(logits).all():
                raise ValueError("Logits chứa NaN/Inf")
            if not math.isfinite(loss.item()):
                raise ValueError("Loss chứa NaN/Inf")

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int(
                (logits.argmax(dim=1) == labels).sum().item()
            )
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader đánh giá rỗng")

    return {
        "loss": float(total_loss / total_samples),
        "accuracy": float(total_correct / total_samples),
    }
def collect_predictions(model, loader, device):
    """Thu dự đoán theo đúng thứ tự dataset của loader."""
    import torch
    from torch.utils.data import SequentialSampler

    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("collect_predictions yêu cầu loader không shuffle")
    if loader.drop_last:
        raise ValueError("collect_predictions yêu cầu drop_last=False")

    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)

            if tuple(logits.shape) != (labels.shape[0], 10):
                raise ValueError("Model phải trả logits [B, 10]")
            if not torch.isfinite(logits).all():
                raise ValueError("Logits chứa NaN/Inf")

            true_labels.extend(labels.detach().cpu().tolist())
            predicted_labels.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

    if not true_labels:
        raise ValueError("DataLoader dự đoán rỗng")
    if len(true_labels) != len(loader.dataset):
        raise ValueError("Loader chưa duyệt đủ dataset")

    return {
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "indices": list(range(len(true_labels))),
    }

def evaluate_final(checkpoint_path, device):
    """Đánh giá checkpoint đã chọn; lưu output cạnh checkpoint."""
    import csv
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from .utils import project_path, save_json
    from .predict import load_model
    from .visualize import plot_predictions

    checkpoint_path = project_path(checkpoint_path).resolve()
    model, metadata = load_model(checkpoint_path, device)

    preprocessing = metadata["preprocessing"]
    if preprocessing["transform"] != "to_tensor":
        raise ValueError("Checkpoint phải dùng transform='to_tensor'")

    # Chỉ đọc tập test gốc; không cần chia lại train/validation.
    dataset = datasets.FashionMNIST(
        root=str(project_path(preprocessing["data_root"])),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    class_names = metadata["class_names"]
    if list(dataset.classes) != class_names:
        raise ValueError("Thứ tự tên lớp dataset khác checkpoint")

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    metrics = evaluate(
        model, loader, torch.nn.CrossEntropyLoss(), device
    )
    predictions = collect_predictions(model, loader, device)

    output_dir = checkpoint_path.parent
    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"

    save_json(metrics_path, {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "num_samples": len(dataset),
        "run_id": metadata["run_id"],
        "epoch": metadata["epoch"],
        "checkpoint_path": str(checkpoint_path),
    })

    with predictions_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "test_index", "true_label", "predicted_label"
        ])
        writer.writerows(zip(
            predictions["indices"],
            predictions["true_labels"],
            predictions["predicted_labels"],
        ))

    image_path = plot_predictions(
        model,
        dataset,
        predictions["indices"][:12],
        class_names,
        device,
        output_dir / "predictions.png",
    )

    wrong_indices = [
        index
        for index, actual, predicted in zip(
            predictions["indices"],
            predictions["true_labels"],
            predictions["predicted_labels"],
        )
        if actual != predicted
    ]

    result = {
        "test_metrics_path": str(metrics_path),
        "test_predictions_path": str(predictions_path),
        "predictions_path": image_path,
    }

    if wrong_indices:
        result["misclassified_path"] = plot_predictions(
            model,
            dataset,
            wrong_indices[:12],
            class_names,
            device,
            output_dir / "misclassified.png",
        )

    return result

import torch



def evaluate_model(model, dataloader, device):
    """Giữ tương thích tên hàm cũ; dùng chung evaluate()."""
    import torch

    metrics = evaluate(
        model,
        dataloader,
        torch.nn.CrossEntropyLoss(),
        device,
    )

    return {
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
    }