"""TV4 — chỉ tạo hình từ dữ liệu thực tế."""

def plot_history(history_csv, output_path):
    """Vẽ train_loss/val_loss theo epoch, nhãn trục và legend, lưu PNG rồi close.
    Return str đường dẫn PNG.
    """
    raise NotImplementedError("TV4: triển khai plot_history")

def plot_predictions(model, dataset, indices, class_names, device, output_path):
    """Mỗi ảnh hiển thị Actual/Predicted; eval/no_grad; đúng cặp ảnh và nhãn.
    Return str đường dẫn PNG. Nếu chỉ vẽ lỗi phải chọn từ lỗi thực tế.
    """
    raise NotImplementedError("TV4: triển khai plot_predictions")
    """TV4 - Đánh giá mô hình FashionMNIST."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def evaluate_model(model, test_loader, device, class_names=None):
    """
    Đánh giá mô hình trên tập dữ liệu kiểm thử.

    Args:
        model: Mô hình PyTorch đã huấn luyện.
        test_loader: DataLoader của tập kiểm thử.
        device: cpu hoặc cuda.
        class_names: Danh sách tên 10 lớp.

    Returns:
        Dictionary chứa các chỉ số đánh giá.
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images = batch["images"]
                labels = batch["labels"]
            else:
                images, labels = batch[:2]

            images = images.to(device)
            labels = labels.to(device).long()

            logits = model(images)
            loss = F.cross_entropy(
                logits, labels, reduction="sum"
            )

            predictions = logits.argmax(dim=1)

            total_loss += loss.item()
            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    if total_samples == 0:
        raise ValueError("Tập kiểm thử không có dữ liệu.")

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    accuracy = total_correct / total_samples
    average_loss = total_loss / total_samples

    num_classes = (
        len(class_names) if class_names is not None else
        int(max(labels.max(), predictions.max()).item()) + 1
    )

    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.int64
    )

    for true_label, predicted_label in zip(labels, predictions):
        confusion[true_label, predicted_label] += 1

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(num_classes):
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    results = {
        "test_loss": average_loss,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "macro_precision": sum(precision_scores) / num_classes,
        "macro_recall": sum(recall_scores) / num_classes,
        "macro_f1": sum(f1_scores) / num_classes,
        "confusion_matrix": confusion.tolist(),
        "num_samples": total_samples,
    }

    print("\n===== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH =====")
    print(f"Số mẫu kiểm thử : {total_samples}")
    print(f"Test loss       : {average_loss:.4f}")
    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print(f"Macro precision : {results['macro_precision']:.4f}")
    print(f"Macro recall    : {results['macro_recall']:.4f}")
    print(f"Macro F1-score  : {results['macro_f1']:.4f}")

    return results


def save_evaluation(results, output_path):
    """Lưu kết quả đánh giá thành file JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Đã lưu kết quả tại: {output_path}")

"""TV4 - Trực quan hóa kết quả đánh giá FashionMNIST."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    output_path="outputs/confusion_matrix.png",
):
    """Vẽ và lưu ma trận nhầm lẫn."""

    matrix = np.asarray(confusion_matrix)

    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError(
            "Kích thước ma trận phải khớp với số lớp."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Nhãn dự đoán",
        ylabel="Nhãn thực tế",
        title="FashionMNIST - Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")


def plot_metrics(
    results,
    output_path="outputs/evaluation_metrics.png",
):
    """Vẽ biểu đồ các chỉ số đánh giá."""

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    metric_values = [
        results["accuracy"],
        results["macro_precision"],
        results["macro_recall"],
        results["macro_f1"],
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(metric_names, metric_values)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("FashionMNIST - Evaluation Metrics")

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")

def plot_predictions(
    images,
    true_labels,
    predicted_labels,
    class_names,
    output_path="outputs/predictions.png",
    max_images=12,
):
    """Hiển thị ảnh FashionMNIST và nhãn dự đoán."""

    from pathlib import Path
    import matplotlib.pyplot as plt
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()

    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().tolist()

    if isinstance(predicted_labels, torch.Tensor):
        predicted_labels = (
            predicted_labels.detach().cpu().tolist()
        )

    n = min(max_images, len(images))

    if n == 0:
        raise ValueError("Không có ảnh để hiển thị.")

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        image = images[i].squeeze().numpy()

        true_name = class_names[int(true_labels[i])]
        pred_name = class_names[int(predicted_labels[i])]

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"Thật: {true_name}\nDự đoán: {pred_name}"
        )
        ax.axis("off")

    fig.suptitle("FashionMNIST - Prediction Results")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu hình ảnh dự đoán: {output_path}")
    
