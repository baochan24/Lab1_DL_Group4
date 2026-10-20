"""TV1 — giữ nguyên giao diện; tham khảo docs/TEAM_CONTRACT.md."""

import os
import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt

# Danh sách 10 nhãn chuẩn của FashionMNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def prepare_data(config):
    # 1. Đọc đường dẫn và tham số từ config
    data_root = config.get("data_root", "data")
    output_root = config.get("output_root", "outputs")
    split_path = config.get("split_indices_path", os.path.join(output_root, "split_indices.json"))
    summary_path = config.get("data_summary_path", os.path.join(output_root, "data_summary.json"))
    dist_path = config.get("class_distribution_path", os.path.join(output_root, "class_distribution.csv"))
    sample_img_path = config.get("sample_images_path", os.path.join(output_root, "sample_images.png"))

    train_size = config.get("train_size", 54000)
    val_size = config.get("val_size", 6000)
    seed = config.get("seed", 42)
    batch_size = config.get("batch_size", 64)

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    # 2. Tải dữ liệu từ Torchvision bằng transform ToTensor
    transform = transforms.ToTensor()
    full_train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    total_dataset_size = len(full_train_dataset)

    # 3. Tạo hoặc Đọc lại split indices (Không ghi đè nếu file đã tồn tại)
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        train_indices = split_data["train_indices"]
        val_indices = split_data["val_indices"]
    else:
        # Khởi tạo generator với seed cố định để chia ngẫu nhiên
        g = torch.Generator().manual_seed(seed)
        permuted_indices = torch.randperm(total_dataset_size, generator=g).tolist()

        train_indices = permuted_indices[:train_size]
        val_indices = permuted_indices[train_size : train_size + val_size]

        split_data = {
            "seed": seed,
            "dataset_size": total_dataset_size,
            "train_indices": train_indices,
            "val_indices": val_indices
        }
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=4)

    # 4. Xác minh chỉ số đầy đủ và không giao nhau
    set_train = set(train_indices)
    set_val = set(val_indices)
    is_disjoint = len(set_train.intersection(set_val)) == 0
    total_split_count = len(train_indices) + len(val_indices)

    assert is_disjoint, "Lỗi: Tập train_indices và val_indices bị giao nhau!"
    assert total_split_count <= total_dataset_size, f"Lỗi: Tổng số chỉ số chia ({total_split_count}) khác tổng dữ liệu gốc ({total_dataset_size})!"

    # 5. Kiểm tra thông số batch và tạo data_summary.json
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    check_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    first_images, first_labels = next(iter(check_loader))

    summary_data = {
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_dataset),
        "image_shape": list(first_images.shape[1:]),
        "batch_image_shape": list(first_images.shape),
        "batch_label_shape": list(first_labels.shape),
        "pixel_min": float(first_images.min()),
        "pixel_max": float(first_images.max()),
        "image_dtype": str(first_images.dtype),
        "label_dtype": str(first_labels.dtype),
        "split_is_disjoint": is_disjoint
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # 6. Đếm số mẫu từng lớp và lưu class_distribution.csv
    # Lấy nhãn gốc từ .targets
    targets_train_val = full_train_dataset.targets
    targets_test = test_dataset.targets

    train_labels = targets_train_val[train_indices].tolist()
    val_labels = targets_train_val[val_indices].tolist()
    test_labels = targets_test.tolist()

    distribution_list = []
    for c_id in range(len(CLASS_NAMES)):
        distribution_list.append({
            "class_id": c_id,
            "class_name": CLASS_NAMES[c_id],
            "train_count": train_labels.count(c_id),
            "val_count": val_labels.count(c_id),
            "test_count": test_labels.count(c_id)
        })

    df_dist = pd.DataFrame(distribution_list)
    df_dist.to_csv(dist_path, index=False)

    # 7. Trực quan hóa 16 ảnh mẫu cố định
    plt.figure(figsize=(10, 10))
    for i in range(16):
        img, label = train_subset[i]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img.squeeze().numpy(), cmap="gray", interpolation="nearest")
        plt.title(f"{label}: {CLASS_NAMES[label]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(sample_img_path, dpi=150)
    plt.close()

    # 8. Trả về dict chứa danh sách đường dẫn output thực tế
    return {
        "split_indices": split_path,
        "data_summary": summary_path,
        "class_distribution": dist_path,
        "sample_images": sample_img_path
    }


def build_dataloaders(config):
    """
    Dùng split đã chuẩn bị; trả train_loader, val_loader, test_loader, class_names.
    Transform ToTensor; train shuffle với Generator có seed, val/test không shuffle.
    Không tự chia split mới theo từng run. Lấy đường dẫn bằng project_path.
    """
    data_root = config.get("data_root", "data")
    output_root = config.get("output_root", "outputs")
    split_path = config.get("split_indices_path", os.path.join(output_root, "split_indices.json"))
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 0)
    seed = config.get("seed", 42)

    transform = transforms.ToTensor()

    # Tải dataset
    full_train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    # Đọc lại file split_indices.json đã chuẩn bị trước đó
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Không tìm thấy file split indices tại '{split_path}'. Hãy gọi prepare_data(config) trước!")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # Tạo Generator với seed cố định cho việc shuffle tập train
    g = torch.Generator()
    g.manual_seed(seed)

    # Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, CLASS_NAMES


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("Bat dau chuan bi du lieu FashionMNIST...")
    print("=" * 60)

    config = {
        "data_root": "data",
        "output_root": "outputs",
        "seed": 42,
        "batch_size": 64,
        "num_workers": 0
    }

    # 1. Chay chuan bi du lieu va xuat thong ke, anh mau
    outputs = prepare_data(config)
    print("\n[OK] Da chuan bi du lieu thanh cong! Cac file outputs:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")

    # 2. Tao DataLoaders de kiem tra
    train_loader, val_loader, test_loader, classes = build_dataloaders(config)
    print("\n[OK] Tao DataLoaders thanh cong:")
    print(f"  - So batch Train: {len(train_loader)} (tong {len(train_loader.dataset)} mau)")
    print(f"  - So batch Val  : {len(val_loader)} (tong {len(val_loader.dataset)} mau)")
    print(f"  - So batch Test : {len(test_loader)} (tong {len(test_loader.dataset)} mau)")
    print(f"  - Danh sach 10 nhan: {classes}")
    print("=" * 60)
