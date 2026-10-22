"""TV4 — metrics và test cuối; không import train.py."""

def evaluate(model, loader, loss_fn, device):
    """model.eval + no_grad; không cập nhật trọng số.
    Return {loss: float, accuracy: float}; trung bình theo số mẫu, accuracy 0–1.
    Caller phải gọi lại model.train trước epoch train tiếp theo.
    """
    raise NotImplementedError("TV4: triển khai evaluate sớm cho TV3 sử dụng")

def collect_predictions(model, loader, device):
    """Return {true_labels: list[int], predicted_labels: list[int], indices: list[int]}.
    Contract chỉ hỗ trợ loader không shuffle cho hàm này; indices là thứ tự trong
    dataset loader đang dùng. Với test_dataset gốc: đúng test_index gốc.
    """
    raise NotImplementedError("TV4: triển khai collect_predictions")

def evaluate_final(checkpoint_path, device):
    """Chỉ gọi sau khi TV5 chọn run bằng validation.
    Dùng load_model từ predict, lấy data_root từ metadata/preprocessing.
    Xuất final/test_metrics.json, test_predictions.csv và ảnh Actual/Predicted.
    Return dict đường dẫn output. Không tune dựa trên test.
    """
    raise NotImplementedError("TV4: triển khai evaluate_final sau khi chốt mô hình")
    """Đánh giá mô hình MLP trên tập test."""

import torch
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if not all_labels:
        raise ValueError("Tập dữ liệu đánh giá đang trống.")

    accuracy = accuracy_score(all_labels, all_predictions)
    average_loss = total_loss / len(all_labels)

    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    print(
        classification_report(
            all_labels,
            all_predictions,
            zero_division=0
        )
    )

    return {
        "test_loss": average_loss,
        "test_accuracy": accuracy,
    }
