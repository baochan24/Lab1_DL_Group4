import csv
import json
from pathlib import Path
import math
from .utils import project_path, validate_config, save_json
def summarize_and_select(run_dirs, output_root):
    """
    Đọc config/history/metadata từng run; xuất experiment_results.csv.
    Chọn run bằng best_val_accuracy, hòa bằng best_val_loss; nếu vẫn hòa giữ
    run xuất hiện trước trong run_dirs. Ghi final/selection.json (run_id,
    best_epoch, checkpoint_path, val_metrics, criterion).
    Không dùng test. Return dict {results_path, selection_path, checkpoint_path}.
    Không điền dữ liệu giả khi run thất bại/chưa có checkpoint.
    """
    output_root_path = project_path(output_root).resolve()
    final_dir = output_root_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    summarized_runs = []

    for run_dir_input in run_dirs:
        run_dir = project_path(run_dir_input).resolve()
        config_file = run_dir / "config.json"
        metadata_file = run_dir / "metadata.json"
        history_file = run_dir / "history.csv"
        checkpoint_file = run_dir / "best_model.pth"

        # Bỏ qua run chưa chạy xong, thất bại hoặc thiếu checkpoint/history (không điền dữ liệu giả)
        if not (history_file.exists() and config_file.exists() and checkpoint_file.exists()):
            continue

        # Đọc cấu hình
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            validate_config(config)

        # Đọc metadata nếu có
                # Chỉ xét run đã train xong.
        if not metadata_file.is_file():
            continue

        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Metadata không phải object: {metadata_file}"
            )

        if metadata.get("status") not in ("trained", "completed"):
            continue

        if metadata.get("training_status", "trained") != "trained":
            continue

        # Đọc lịch sử huấn luyện
        epochs_data = []
        with open(history_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs_data.append({
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "val_loss": float(row["val_loss"]),
                    "val_accuracy": float(row["val_accuracy"])
                })
        if not epochs_data:
            continue

        # Kiểm tra đủ epoch, đúng thứ tự và không trùng.
        expected_epochs = list(range(1, config["epochs"] + 1))
        actual_epochs = [row["epoch"] for row in epochs_data]

        if actual_epochs != expected_epochs:
            raise ValueError(
                f"History thiếu/lặp/sai thứ tự epoch: {run_dir}"
            )

        # Kiểm tra metric hữu hạn và hợp lệ.
        for row in epochs_data:
            for key in (
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ):
                value = row[key]
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Metric không hợp lệ: {run_dir}, "
                        f"epoch={row['epoch']}, {key}={value}"
                    )

            if row["train_accuracy"] > 1 or row["val_accuracy"] > 1:
                raise ValueError(
                    f"Accuracy ngoài [0, 1]: {run_dir}"
                )

        # Chọn accuracy cao nhất, hòa thì loss thấp nhất.
        # Nếu cả hai hòa, giữ epoch xuất hiện trước.
        best_epoch_data = max(
            epochs_data,
            key=lambda x: (
                x["val_accuracy"],
                -x["val_loss"],
                -x["epoch"],
            ),
        )


        run_info = {
            "run_id": str(config.get("run_id", run_dir.name)),
            "hidden_units": int(config.get("hidden_units", 128)),
            "learning_rate": float(config.get("learning_rate", 0.001)),
            "batch_size": int(config.get("batch_size", 64)),
            "epochs": int(config.get("epochs", 10)),
            "seed": int(config.get("seed", 42)),
            "best_epoch": int(best_epoch_data["epoch"]),
            "best_val_accuracy": float(best_epoch_data["val_accuracy"]),
            "best_val_loss": float(best_epoch_data["val_loss"]),
            "duration_seconds": metadata.get("duration_seconds", "N/A"),
            "device": str(metadata.get("device", config.get("device", "auto"))),
            "commit_sha": str(metadata.get("commit_sha", "N/A")),
            "checkpoint_path": str(checkpoint_file.resolve())
        }
        summarized_runs.append(run_info)

    # Nếu không có run nào thành công thì báo lỗi, không điền dữ liệu giả
    if not summarized_runs:
        raise FileNotFoundError("Không tìm thấy run hợp lệ nào có đủ config.json, history.csv và best_model.pth.")

    # 1. Xuất file experiment_results.csv
    results_csv_path = output_root_path / "experiment_results.csv"
    csv_headers = [
        "run_id", "hidden_units", "learning_rate", "batch_size", "epochs",
        "seed", "best_epoch", "best_val_accuracy", "best_val_loss",
        "duration_seconds", "device", "commit_sha", "checkpoint_path"
    ]


    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for run in summarized_runs:
            writer.writerow({k: run[k] for k in csv_headers})

    # 2. Chọn run tốt nhất:
    # - best_val_accuracy cao nhất
    # - hòa thì best_val_loss thấp nhất
    # - hòa tiếp giữ run xuất hiện trước trong run_dirs (dùng -item[0] làm tie-breaker)
    best_run = max(
        enumerate(summarized_runs),
        key=lambda item: (item[1]["best_val_accuracy"], -item[1]["best_val_loss"], -item[0])
    )[1]

    # 3. Ghi file final/selection.json đúng cấu trúc yêu cầu
    selection_data = {
        "run_id": best_run["run_id"],
        "best_epoch": best_run["best_epoch"],
        "checkpoint_path": best_run["checkpoint_path"],
        "val_metrics": {
            "loss": best_run["best_val_loss"],
            "accuracy": best_run["best_val_accuracy"]
        },
        "criterion": "highest best_val_accuracy, lowest best_val_loss"
    }

    selection_json_path = final_dir / "selection.json"
    save_json(selection_json_path, selection_data)
    # 4. Trả về dict kết quả
    return {
        "results_path": str(results_csv_path.resolve()),
        "selection_path": str(selection_json_path.resolve()),
        "checkpoint_path": str(Path(best_run["checkpoint_path"]).resolve())
    }