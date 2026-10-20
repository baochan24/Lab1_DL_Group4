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
