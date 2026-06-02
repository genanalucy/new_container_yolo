from __future__ import annotations

import csv
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/hzh/container_yolo")
RF_ROOT = ROOT / "public_datasets" / "roboflow_container_number_detection_v2_yolov8"
B_ROOT = ROOT / "container_v1i_yolov11_b"
MERGED_ROOT = ROOT / "merged_container_number_yolo_b_rf"
AUDIT_ROOT = ROOT / "public_datasets" / "roboflow_container_number_detection_v2_yolov8_audit"
TRAIN_SCRIPT = ROOT / "train_b_rf_960.py"
TRAIN_SESSION = "container_b_rf_960"
DOWNLOAD_SESSION = "roboflow_download_v2_yolov8"
BONLY_1280_SESSION = "container_bonly_1280"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASS_NAMES = {"container_number", "container number", "container-number", "container_number_region", "number", "objects"}


def tmux_has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def wait_for_download() -> None:
    while True:
        session_running = tmux_has_session(DOWNLOAD_SESSION)
        data_yaml = RF_ROOT / "data.yaml"
        has_split = any((RF_ROOT / split / "images").exists() for split in ["train", "valid", "test"])
        if not session_running and data_yaml.exists() and has_split:
            return
        print(
            f"waiting_for_roboflow_download session_running={session_running} "
            f"data_yaml={data_yaml.exists()} has_split={has_split}",
            flush=True,
        )
        time.sleep(60)


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".txt")


def parse_data_yaml_names() -> list[str]:
    data_yaml = RF_ROOT / "data.yaml"
    lines = data_yaml.read_text(errors="replace").splitlines()
    stripped_lines = [line.strip() for line in lines]
    for index, line in enumerate(stripped_lines):
        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            if value.startswith("[") and value.endswith("]"):
                return [part.strip().strip("'\"") for part in value.strip("[]").split(",") if part.strip()]
            names = []
            for child in stripped_lines[index + 1:]:
                if not child.startswith("-"):
                    break
                names.append(child[1:].strip().strip("'\""))
            return names
    return []


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"bad label format {path}: {raw_line}")
        cls = int(float(parts[0]))
        x, y, w, h = [float(value) for value in parts[1:]]
        if not (0 <= cls and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"bad label value {path}: {raw_line}")
        rows.append((cls, x, y, w, h))
    return rows


def draw_sample(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for cls, x, y, w, h in read_label(label_path):
        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height
        draw.rectangle((x1, y1, x2, y2), outline="red", width=max(2, width // 400))
        draw.text((x1, max(0, y1 - 14)), str(cls), fill="red")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((640, 640))
    image.save(output_path, quality=90)


def make_contact_sheet(samples: list[Path], output_path: Path, columns: int = 5, cell_size: tuple[int, int] = (360, 270)) -> None:
    if not samples:
        return
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_size[0], rows * cell_size[1]), "white")
    for index, sample in enumerate(samples):
        image = Image.open(sample).convert("RGB")
        image.thumbnail(cell_size)
        x = (index % columns) * cell_size[0] + (cell_size[0] - image.width) // 2
        y = (index // columns) * cell_size[1] + (cell_size[1] - image.height) // 2
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def audit_dataset() -> dict[str, object]:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    names = parse_data_yaml_names()
    class_counter: Counter[int] = Counter()
    image_count_by_split = {}
    label_count_by_split = {}
    boxes_by_split = {}
    missing_labels = []
    bad_files = []
    boxes_per_image = []
    small_box_count = 0
    suspicious_single_character_like = 0

    for split in ["train", "valid", "test"]:
        images = image_files(RF_ROOT / split / "images")
        labels_dir = RF_ROOT / split / "labels"
        image_count_by_split[split] = len(images)
        label_count_by_split[split] = len(label_files(labels_dir))
        split_boxes = 0
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(str(image_path))
                continue
            try:
                rows = read_label(label_path)
            except Exception as exc:
                bad_files.append(f"{label_path}: {exc}")
                continue
            split_boxes += len(rows)
            boxes_per_image.append(len(rows))
            for cls, _x, _y, w, h in rows:
                class_counter[cls] += 1
                area = w * h
                aspect = max(w / h, h / w)
                if area < 0.0003:
                    small_box_count += 1
                if area < 0.003 and aspect > 1.8:
                    suspicious_single_character_like += 1
        boxes_by_split[split] = split_boxes

    total_images = sum(image_count_by_split.values())
    total_boxes = sum(boxes_by_split.values())
    summary: dict[str, object] = {
        "names": names,
        "image_count_by_split": image_count_by_split,
        "label_count_by_split": label_count_by_split,
        "boxes_by_split": boxes_by_split,
        "class_counter": dict(class_counter),
        "missing_label_count": len(missing_labels),
        "bad_file_count": len(bad_files),
        "total_images": total_images,
        "total_boxes": total_boxes,
        "max_boxes_per_image": max(boxes_per_image) if boxes_per_image else 0,
        "mean_boxes_per_image": round(sum(boxes_per_image) / len(boxes_per_image), 4) if boxes_per_image else 0,
        "small_box_count": small_box_count,
        "suspicious_single_character_like": suspicious_single_character_like,
    }

    all_images = []
    for split in ["train", "valid", "test"]:
        all_images.extend(image_files(RF_ROOT / split / "images"))
    random.seed(20260602)
    sampled_images = random.sample(all_images, min(100, len(all_images)))
    visualized = []
    for image_path in sampled_images:
        split = image_path.parents[1].name
        label_path = RF_ROOT / split / "labels" / f"{image_path.stem}.txt"
        output_path = AUDIT_ROOT / "samples" / f"{split}_{image_path.name}"
        draw_sample(image_path, label_path, output_path)
        visualized.append(output_path)
    make_contact_sheet(visualized[:50], AUDIT_ROOT / "contact_sheet_01.jpg")
    make_contact_sheet(visualized[50:100], AUDIT_ROOT / "contact_sheet_02.jpg")

    classes_ok = set(class_counter.keys()).issubset({0}) and bool(names) and len(names) == 1
    name_ok = names[0].strip().lower() in EXPECTED_CLASS_NAMES if names else False
    counts_ok = total_images >= 1000 and total_boxes >= 1000
    labels_ok = len(missing_labels) == 0 and len(bad_files) == 0
    single_char_risk_ok = suspicious_single_character_like / max(1, total_boxes) < 0.2
    passed = bool(classes_ok and name_ok and counts_ok and labels_ok and single_char_risk_ok)
    summary.update({
        "passed_basic_audit": passed,
        "classes_ok": classes_ok,
        "name_ok": name_ok,
        "counts_ok": counts_ok,
        "labels_ok": labels_ok,
        "single_char_risk_ok": single_char_risk_ok,
    })

    with (AUDIT_ROOT / "summary.md").open("w") as file:
        file.write("# Roboflow Container Number Detection v2 审核摘要\n\n")
        for key, value in summary.items():
            file.write(f"- {key}: {value}\n")
        file.write("\n## 说明\n\n")
        file.write("该审核是自动基础审核，只检查类别、标签格式、数量和明显单字符风险；contact sheet 仍需人工复核语义。\n")
    (AUDIT_ROOT / "missing_labels.txt").write_text("\n".join(missing_labels))
    (AUDIT_ROOT / "bad_files.txt").write_text("\n".join(bad_files))
    return summary


def copy_with_prefix(image_path: Path, label_path: Path, output_image_dir: Path, output_label_dir: Path, prefix: str, force_class_zero: bool) -> None:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_image_dir / f"{prefix}_{image_path.name}")
    output_label = output_label_dir / f"{prefix}_{image_path.stem}.txt"
    if force_class_zero:
        rows = read_label(label_path)
        with output_label.open("w") as file:
            for _cls, x, y, w, h in rows:
                file.write(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n")
    else:
        shutil.copy2(label_path, output_label)


def merge_dataset() -> None:
    if MERGED_ROOT.exists():
        backup = MERGED_ROOT.with_name(f"{MERGED_ROOT.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
        MERGED_ROOT.rename(backup)
    for split in ["train", "val"]:
        (MERGED_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for b_split, merged_split in [("train", "train"), ("valid", "val")]:
        for image_path in image_files(B_ROOT / b_split / "images"):
            label_path = B_ROOT / b_split / "labels" / f"{image_path.stem}.txt"
            copy_with_prefix(image_path, label_path, MERGED_ROOT / "images" / merged_split, MERGED_ROOT / "labels" / merged_split, f"b_{b_split}", False)

    for rf_split, merged_split in [("train", "train"), ("valid", "val")]:
        for image_path in image_files(RF_ROOT / rf_split / "images"):
            label_path = RF_ROOT / rf_split / "labels" / f"{image_path.stem}.txt"
            copy_with_prefix(image_path, label_path, MERGED_ROOT / "images" / merged_split, MERGED_ROOT / "labels" / merged_split, f"rf_{rf_split}", True)

    eval_root = ROOT / "eval_sets" / "roboflow_container_number_test"
    for subdir in ["images", "labels"]:
        (eval_root / subdir).mkdir(parents=True, exist_ok=True)
    for image_path in image_files(RF_ROOT / "test" / "images"):
        label_path = RF_ROOT / "test" / "labels" / f"{image_path.stem}.txt"
        copy_with_prefix(image_path, label_path, eval_root / "images", eval_root / "labels", "rf_test", True)

    (MERGED_ROOT / "data.yaml").write_text(
        "path: /home/hzh/container_yolo/merged_container_number_yolo_b_rf\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['container_number_region']\n"
    )

    rows = []
    for split in ["train", "val"]:
        rows.append({
            "split": split,
            "images": len(image_files(MERGED_ROOT / "images" / split)),
            "labels": len(label_files(MERGED_ROOT / "labels" / split)),
        })
    with (MERGED_ROOT / "merge_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "images", "labels"])
        writer.writeheader()
        writer.writerows(rows)


def write_train_script() -> None:
    TRAIN_SCRIPT.write_text(
        "from ultralytics import YOLO\n\n"
        "model = YOLO('/home/hzh/container_yolo/yolo11m.pt')\n\n"
        "model.train(\n"
        "    data='/home/hzh/container_yolo/merged_container_number_yolo_b_rf/data.yaml',\n"
        "    epochs=100,\n"
        "    imgsz=960,\n"
        "    batch=32,\n"
        "    device='0,1,2,3',\n"
        "    workers=16,\n"
        "    project='/home/hzh/container_yolo/runs',\n"
        "    name='region_b_rf_yolo11m_960_100e',\n"
        "    exist_ok=True,\n"
        "    patience=30,\n"
        ")\n"
    )


def start_training_after_bonly_1280() -> None:
    while tmux_has_session(BONLY_1280_SESSION):
        print(f"waiting_for_bonly_1280_to_finish session={BONLY_1280_SESSION}", flush=True)
        time.sleep(300)
    if tmux_has_session(TRAIN_SESSION):
        print(f"training_session_already_running={TRAIN_SESSION}", flush=True)
        return
    log_path = ROOT / "runs" / "region_b_rf_yolo11m_960_100e_train.log"
    command = f"cd {ROOT} && python3 {TRAIN_SCRIPT} 2>&1 | tee {log_path}"
    subprocess.run(["tmux", "new-session", "-d", "-s", TRAIN_SESSION, command], check=True)
    print(f"started_training_session={TRAIN_SESSION}", flush=True)


def main() -> None:
    wait_for_download()
    summary = audit_dataset()
    print(f"audit_summary={summary}", flush=True)
    if not summary["passed_basic_audit"]:
        raise SystemExit(f"Roboflow basic audit failed; see {AUDIT_ROOT / 'summary.md'}")
    merge_dataset()
    write_train_script()
    start_training_after_bonly_1280()


if __name__ == "__main__":
    main()
