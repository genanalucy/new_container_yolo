# 集装箱编号区域检测 YOLO 模型

本仓库用于训练和部署一个单类别 YOLO 检测模型，目标只检测集装箱箱体上的编号区域。

不检测车牌、车辆编号、港口设备编号、建筑编号、路牌编号、Logo 或其它非集装箱文字数字。

## 1. 仓库内容

```text
models/
  best_bonly_1280.pt                         # 已可用模型，约 39 MB
scripts/
  predict_container_number.py                # 本地推理脚本
  train_bonly_1280.py                        # B 数据集高分辨率训练脚本
  train_b_rf_960.py                          # B + Roboflow 合并集训练脚本
  auto_audit_merge_train_rf.py               # Roboflow 审核、合并、训练启动脚本
docs/training_results/
  region_b_yolo11m_1280_100e_results.csv     # B-only-1280 训练结果
  region_b_rf_yolo11m_960_100e_results.csv   # B+Roboflow-960 训练结果
requirements.txt
```

当前随仓库提供的可直接使用模型：

```text
models/best_bonly_1280.pt
```

该模型来自 YOLO11m，在 B 数据集上使用 `imgsz=1280` 训练，适合先做本地推理验证。

## 2. 环境准备

建议使用 Python 3.10 或 3.11。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

训练脚本会在启动时检查 `ultralytics`，如果当前 Python 环境缺少依赖，会提示先运行上面的安装命令。

如果在 Mac 上只做 CPU 推理，也可以直接安装：

```bash
python3 -m pip install ultralytics pillow
```

## 3. 本地推理使用

预测单张图片：

```bash
python3 scripts/predict_container_number.py /path/to/image.jpg
```

预测一个图片文件夹：

```bash
python3 scripts/predict_container_number.py /path/to/images
```

推荐远景图片参数：

```bash
python3 scripts/predict_container_number.py /path/to/image_or_folder \
  --imgsz 1280 \
  --conf 0.05 \
  --iou 0.5 \
  --device cpu
```

如果 Mac 支持 MPS，可以尝试：

```bash
python3 scripts/predict_container_number.py /path/to/image_or_folder --device mps
```

输出默认保存到：

```text
outputs/predict_results/container_number_region/
```

输出内容包括：

- 画框后的图片。
- YOLO 格式标签文件。
- 带置信度的预测文本。

## 4. 数据处理流程

### 4.1 原始 B 数据集

B 数据集是当前主数据集，类别统一为单类：

```yaml
nc: 1
names: ['container_number_region']
```

训练前需要确认：

1. 图片和标签按 YOLO 格式组织。
2. 每张图片的标签文件与图片同名。
3. 标签格式为：

```text
class x_center y_center width height
```

4. 所有类别编号统一为 `0`。
5. 标注框语义必须是完整集装箱编号区域，不是单字符、Logo、车辆编号或其它文字。

服务器上的 B 数据集路径为：

```text
/home/hzh/container_yolo/container_v1i_yolov11_b/data_fixed.yaml
```

### 4.2 Roboflow 公开数据集审核

使用的公开数据集为 Roboflow Container Number Detection v2，导出格式为 YOLOv8。

审核脚本：

```text
scripts/auto_audit_merge_train_rf.py
```

审核内容包括：

1. 读取 `data.yaml` 的类别名。
2. 检查 train、valid、test 图片数量和标签数量。
3. 检查标签格式是否合法。
4. 统计类别分布，确认只存在单类别。
5. 统计框数量和每图框数。
6. 检查疑似单字符标注风险。
7. 抽样生成 contact sheet，人工复核标注语义。
8. 审核通过后统一类别为 `0 container_number_region`。

自动审核只负责基础质量检查，contact sheet 仍需要人工确认是否存在非集装箱编号误标。

### 4.3 B + Roboflow 合并策略

合并目录：

```text
/home/hzh/container_yolo/merged_container_number_yolo_b_rf
```

合并规则：

1. B 数据集 train 加入合并 train。
2. B 数据集 valid 加入合并 val。
3. Roboflow train 加入合并 train。
4. Roboflow valid 加入合并 val。
5. Roboflow test 单独保留为外部测试集，不混入训练集。
6. 所有 Roboflow 标签类别强制改为 `0`。
7. 复制文件时增加来源前缀，避免同名覆盖。

合并后的 `data.yaml`：

```yaml
path: /home/hzh/container_yolo/merged_container_number_yolo_b_rf
train: images/train
val: images/val
nc: 1
names: ['container_number_region']
```

## 5. 训练流程

本项目训练分成两条路线：

1. 先训练 `B-only-1280`，只使用原 B 数据集，用更高输入分辨率提升小目标检测能力。
2. 再训练 `B+Roboflow-960`，把审核通过的 Roboflow 公开数据合并进来，提高不同拍摄角度、不同箱体样式下的泛化能力。

两条路线都使用 YOLO11m 作为基础模型，任务类型都是单类别目标检测。

### 5.1 B-only-1280 训练

训练目标：

- 数据只使用 B 数据集。
- 类别只保留 `container_number_region`。
- 输入尺寸提高到 `1280`，用于增强远距离、小编号区域的召回。
- 使用 `patience=30`，如果验证指标长期没有提升，则提前停止。

训练脚本：

```text
scripts/train_bonly_1280.py
```

脚本默认读取仓库内路径：

```text
datasets/container_v1i_yolov11_b/data_fixed.yaml
runs/
```

也可以通过命令行参数覆盖数据、权重、输出目录和训练设备。

关键参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `yolo11m.pt` | YOLO 预训练权重路径或模型名 |
| `--data` | `datasets/container_v1i_yolov11_b/data_fixed.yaml` | B 数据集配置文件 |
| `--project` | `runs` | 训练输出目录 |
| `--name` | `region_b_yolo11m_1280_100e` | 实验名称 |
| `--epochs` | `100` | 最多训练 100 轮 |
| `--imgsz` | `1280` | 高分辨率训练，用于小目标场景 |
| `--batch` | `16` | batch size |
| `--device` | `0` | 训练设备，例如 `0` 或 `0,1,2,3` |
| `--workers` | `8` | 数据加载线程数 |
| `--patience` | `30` | 30 轮无提升则 early stopping |

单机直接启动：

```bash
python3 scripts/train_bonly_1280.py
```

服务器上推荐使用 tmux，避免 SSH 断开导致训练中断：

```bash
tmux new-session -d -s container_bonly_1280 \
  'cd /home/hzh/container_yolo && python3 scripts/train_bonly_1280.py --data /home/hzh/container_yolo/container_v1i_yolov11_b/data_fixed.yaml --project /home/hzh/container_yolo/runs --device 0,1,2,3 --workers 16 2>&1 | tee /home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e_train.log'
```

训练过程监控：

```bash
tmux attach -t container_bonly_1280
```

查看 GPU：

```bash
nvidia-smi
```

查看训练日志：

```bash
tail -f /home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e_train.log
```

查看结果文件：

```bash
ls -lh /home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e
ls -lh /home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e/weights
```

训练完成后重点产物：

```text
/home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e/weights/best.pt
/home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e/weights/last.pt
/home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e/results.csv
/home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e/results.png
```

其中 `best.pt` 是验证集指标最优的权重，本仓库中的模型文件就是从该路径复制而来：

```text
models/best_bonly_1280.pt
```

### 5.2 Roboflow 数据审核与合并

Roboflow 数据不能下载后直接合并，需要先审核标注语义和标签格式。

审核与合并脚本：

```text
scripts/auto_audit_merge_train_rf.py
```

脚本中的关键路径：

```python
ROOT = Path("/home/hzh/container_yolo")
RF_ROOT = ROOT / "public_datasets" / "roboflow_container_number_detection_v2_yolov8"
B_ROOT = ROOT / "container_v1i_yolov11_b"
MERGED_ROOT = ROOT / "merged_container_number_yolo_b_rf"
AUDIT_ROOT = ROOT / "public_datasets" / "roboflow_container_number_detection_v2_yolov8_audit"
TRAIN_SCRIPT = ROOT / "scripts" / "train_b_rf_960.py"
```

审核逻辑会先等待 Roboflow 下载完成：

```python
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
```

标签格式检查代码：

```python
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
```

基础审核会检查：

- `data.yaml` 中的类别名。
- train、valid、test 是否都有图片目录。
- 图片数量、标签数量和框数量。
- 是否有缺失标签。
- 是否有非法 YOLO 标签。
- 类别是否能统一到单类。
- 是否存在较多疑似单字符框。

通过条件代码：

```python
classes_ok = set(class_counter.keys()).issubset({0}) and bool(names) and len(names) == 1
name_ok = names[0].strip().lower() in EXPECTED_CLASS_NAMES if names else False
counts_ok = total_images >= 1000 and total_boxes >= 1000
labels_ok = len(missing_labels) == 0 and len(bad_files) == 0
single_char_risk_ok = suspicious_single_character_like / max(1, total_boxes) < 0.2
passed = bool(classes_ok and name_ok and counts_ok and labels_ok and single_char_risk_ok)
```

脚本还会随机抽样生成可视化审核图：

```python
random.seed(20260602)
sampled_images = random.sample(all_images, min(100, len(all_images)))
for image_path in sampled_images:
    split = image_path.parents[1].name
    label_path = RF_ROOT / split / "labels" / f"{image_path.stem}.txt"
    output_path = AUDIT_ROOT / "samples" / f"{split}_{image_path.name}"
    draw_sample(image_path, label_path, output_path)
    visualized.append(output_path)
make_contact_sheet(visualized[:50], AUDIT_ROOT / "contact_sheet_01.jpg")
make_contact_sheet(visualized[50:100], AUDIT_ROOT / "contact_sheet_02.jpg")
```

审核输出：

```text
/home/hzh/container_yolo/public_datasets/roboflow_container_number_detection_v2_yolov8_audit/summary.md
/home/hzh/container_yolo/public_datasets/roboflow_container_number_detection_v2_yolov8_audit/contact_sheet_01.jpg
/home/hzh/container_yolo/public_datasets/roboflow_container_number_detection_v2_yolov8_audit/contact_sheet_02.jpg
/home/hzh/container_yolo/public_datasets/roboflow_container_number_detection_v2_yolov8_audit/missing_labels.txt
/home/hzh/container_yolo/public_datasets/roboflow_container_number_detection_v2_yolov8_audit/bad_files.txt
```

合并代码会把 B 和 Roboflow 复制到新目录，并统一类别为 `0`：

```python
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
```

合并主流程：

```python
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
```

Roboflow test 不混入训练，而是单独保留为外部测试集：

```python
eval_root = ROOT / "eval_sets" / "roboflow_container_number_test"
for subdir in ["images", "labels"]:
    (eval_root / subdir).mkdir(parents=True, exist_ok=True)
for image_path in image_files(RF_ROOT / "test" / "images"):
    label_path = RF_ROOT / "test" / "labels" / f"{image_path.stem}.txt"
    copy_with_prefix(image_path, label_path, eval_root / "images", eval_root / "labels", "rf_test", True)
```

合并后的 `data.yaml` 由脚本写入：

```python
(MERGED_ROOT / "data.yaml").write_text(
    "path: /home/hzh/container_yolo/merged_container_number_yolo_b_rf\n"
    "train: images/train\n"
    "val: images/val\n"
    "nc: 1\n"
    "names: ['container_number_region']\n"
)
```

完整自动流程入口：

```python
def main() -> None:
    wait_for_download()
    summary = audit_dataset()
    print(f"audit_summary={summary}", flush=True)
    if not summary["passed_basic_audit"]:
        raise SystemExit(f"Roboflow basic audit failed; see {AUDIT_ROOT / 'summary.md'}")
    merge_dataset()
    write_train_script()
    start_training_after_bonly_1280()
```

执行命令：

```bash
python3 scripts/auto_audit_merge_train_rf.py
```

### 5.3 B + Roboflow-960 训练

训练目标：

- 使用 B 数据集和审核通过的 Roboflow 数据。
- Roboflow train 加入训练集，Roboflow valid 加入验证集。
- Roboflow test 保留为外部测试集。
- 输入尺寸使用 `960`，在显存、速度和小目标效果之间折中。

训练脚本：

```text
scripts/train_b_rf_960.py
```

脚本默认读取仓库内路径：

```text
datasets/merged_container_number_yolo_b_rf/data.yaml
runs/
```

也可以通过命令行参数覆盖数据、权重、输出目录和训练设备。

关键参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `yolo11m.pt` | YOLO 预训练权重路径或模型名 |
| `--data` | `datasets/merged_container_number_yolo_b_rf/data.yaml` | B + Roboflow 合并数据集 |
| `--project` | `runs` | 训练输出目录 |
| `--name` | `region_b_rf_yolo11m_960_100e` | 实验名称 |
| `--epochs` | `100` | 最多训练 100 轮 |
| `--imgsz` | `960` | 兼顾速度、显存和小目标表现 |
| `--batch` | `32` | 960 输入尺寸下提高 batch |
| `--device` | `0` | 训练设备，例如 `0` 或 `0,1,2,3` |
| `--workers` | `8` | 数据加载线程数 |
| `--patience` | `30` | 验证指标长期无提升则提前停止 |

单机直接启动：

```bash
python3 scripts/train_b_rf_960.py
```

服务器上使用 tmux：

```bash
tmux new-session -d -s container_b_rf_960 \
  'cd /home/hzh/container_yolo && python3 scripts/train_b_rf_960.py --data /home/hzh/container_yolo/merged_container_number_yolo_b_rf/data.yaml --project /home/hzh/container_yolo/runs --device 0,1,2,3 --workers 16 2>&1 | tee /home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e_train.log'
```

训练过程监控：

```bash
tmux attach -t container_b_rf_960
```

查看训练结果：

```bash
ls -lh /home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e
ls -lh /home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e/weights
```

训练完成后重点产物：

```text
/home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e/weights/best.pt
/home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e/weights/last.pt
/home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e/results.csv
/home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e/results.png
```

### 5.4 训练结果读取方式

每个训练目录下都有 `results.csv`，可用下面代码读取最新 epoch 和最佳 mAP50-95：

```python
from pathlib import Path
import csv

results_path = Path('/home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e/results.csv')
rows = list(csv.DictReader(results_path.open()))
latest = rows[-1]
best = max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))

print('latest epoch:', latest['epoch'])
print('latest mAP50-95:', latest['metrics/mAP50-95(B)'])
print('best epoch:', best['epoch'])
print('best mAP50-95:', best['metrics/mAP50-95(B)'])
```

本仓库也保存了两次训练的结果 CSV：

```text
docs/training_results/region_b_yolo11m_1280_100e_results.csv
docs/training_results/region_b_rf_yolo11m_960_100e_results.csv
```

## 6. 训练中遇到的问题

### 6.1 真实远景图召回不足

最初模型在验证集上 mAP 很高，但对真实港口远景图片不稳定，尤其是远距离小编号。

处理方式：

1. 将 B-only 训练分辨率从常规尺寸提高到 `imgsz=1280`。
2. 推理时使用 `imgsz=1280`。
3. 对远距离图片降低置信度阈值，例如 `conf=0.05`。

### 6.2 合并 A 数据集后指标下降

早期尝试过 B + A 数据训练，但 A 数据转换后存在语义噪声或分布干扰，导致指标下降。

处理方式：

1. 不继续直接扩大 A 数据。
2. 优先使用语义更接近的 Roboflow Container Number Detection 数据集。
3. 新公开数据集进入训练前必须先审核。

### 6.3 公开数据集标注语义不确定

公开数据集可能存在单字符框、非集装箱编号、Logo 或其它编号误标。

处理方式：

1. 自动检查类别、标签格式、数量和疑似单字符风险。
2. 生成 contact sheet 进行人工复核。
3. 只合并审核通过的数据。
4. 类别全部统一为 `container_number_region`。

### 6.4 高分辨率训练显存压力

`imgsz=1280` 对显存要求较高。

处理方式：

1. B-only-1280 使用 `batch=16`。
2. B+Roboflow-960 使用 `batch=32`。
3. 使用 4 张 GPU：`device='0,1,2,3'`。
4. 如果显存不足，先降低 batch，再降低 imgsz。

### 6.5 B-only-1280 没跑满 100 epoch

B-only-1280 在第 87 epoch 提前停止。

原因：

```text
EarlyStopping: Training stopped early as no improvement observed in last 30 epochs.
Best results observed at epoch 57.
```

这是正常 early stopping，不是训练掉线。

## 7. 训练结果

### 7.1 B-only-1280

训练输出：

```text
/home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e
```

随仓库提供模型：

```text
models/best_bonly_1280.pt
```

模型大小：约 39 MB。

训练情况：

- 计划 epoch：100。
- 实际停止：87。
- early stopping patience：30。
- 最佳 epoch：57。

最佳指标：

| 模型 | epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-only-1280 | 57 | 0.98270 | 0.97310 | 0.99420 | 0.73870 |

训练结束日志中的最终验证结果：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.983 | 0.973 | 0.994 | 0.738 |

### 7.2 B + Roboflow-960

训练输出：

```text
/home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e
```

当前观察到的最佳指标：

| 模型 | epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B+Roboflow-960 | 69 | 0.98010 | 0.99172 | 0.99369 | 0.77182 |

说明：

- B+Roboflow-960 的 mAP50-95 高于 B-only-1280。
- 该模型权重约 115 MB，超过 GitHub 普通单文件 100 MB 限制，因此当前未随仓库直接提交。
- 最终是否替换为该模型，需要继续在真实 3 张图片、Roboflow test 和负例挑战集上验证。

## 8. 可复现完整流程

### 步骤 1：克隆仓库

```bash
git clone <repo-url>
cd <repo-dir>
```

### 步骤 2：安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 步骤 3：直接使用已训练模型推理

```bash
python3 scripts/predict_container_number.py /path/to/image_or_folder \
  --imgsz 1280 \
  --conf 0.05 \
  --iou 0.5 \
  --device cpu
```

输出查看：

```text
outputs/predict_results/container_number_region/
```

### 步骤 4：准备训练数据

按 YOLO 格式准备数据：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

`data.yaml` 示例：

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
nc: 1
names: ['container_number_region']
```

标签格式：

```text
0 x_center y_center width height
```

所有坐标必须是 0 到 1 的归一化值。

### 步骤 5：复现 B-only-1280 训练

本地默认数据路径为 `datasets/container_v1i_yolov11_b/data_fixed.yaml`。如果数据放在其它位置，启动时传入 `--data`：

```bash
python3 scripts/train_bonly_1280.py --data /absolute/path/to/your/data.yaml
```

如果只有一张 GPU，传入：

```bash
python3 scripts/train_bonly_1280.py --data /absolute/path/to/your/data.yaml --device 0 --batch 4
```

启动训练：

```bash
python3 scripts/train_bonly_1280.py
```

### 步骤 6：复现 B + Roboflow 数据合并训练

1. 下载 Roboflow Container Number Detection v2 的 YOLO 格式数据。
2. 修改 `scripts/auto_audit_merge_train_rf.py` 中的路径常量。
3. 执行自动审核和合并：

```bash
python3 scripts/auto_audit_merge_train_rf.py
```

4. 确认审核摘要和 contact sheet。
5. 审核通过后运行：

```bash
python3 scripts/train_b_rf_960.py --data /absolute/path/to/merged_container_number_yolo_b_rf/data.yaml
```

### 步骤 7：验证模型

建议至少验证以下集合：

1. B valid。
2. Roboflow test。
3. 真实业务图片。
4. 负例挑战集。
5. 小目标挑战集。

不要只看 mAP。最终模型选择应同时考虑：

- 真实图片检出数。
- 小目标召回率。
- 非集装箱编号误检数。
- Precision / Recall / mAP50 / mAP50-95。

## 9. 后续改进方向

1. 等 B+Roboflow-960 完整训练结束后，导出最终 best.pt。
2. 如果需要提交超过 100 MB 的模型，使用 Git LFS 或发布到 GitHub Release。
3. 建立负例挑战集，重点包含车牌、车辆编号、港口设备编号和建筑编号。
4. 对真实远景图片尝试切片推理。
5. 如果小目标召回仍不足，对比 YOLO11l 或 YOLO11x。
