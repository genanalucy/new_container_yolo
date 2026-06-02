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

### 5.1 B-only-1280 训练

训练脚本：

```text
scripts/train_bonly_1280.py
```

核心参数：

```python
model = YOLO('/home/hzh/container_yolo/yolo11m.pt')

model.train(
    data='/home/hzh/container_yolo/container_v1i_yolov11_b/data_fixed.yaml',
    epochs=100,
    imgsz=1280,
    batch=16,
    device='0,1,2,3',
    workers=16,
    project='/home/hzh/container_yolo/runs',
    name='region_b_yolo11m_1280_100e',
    exist_ok=True,
    patience=30,
)
```

启动命令：

```bash
python3 scripts/train_bonly_1280.py
```

服务器上使用 tmux 运行：

```bash
tmux new-session -d -s container_bonly_1280 \
  'cd /home/hzh/container_yolo && python3 train_bonly_1280.py 2>&1 | tee /home/hzh/container_yolo/runs/region_b_yolo11m_1280_100e_train.log'
```

### 5.2 B + Roboflow-960 训练

训练脚本：

```text
scripts/train_b_rf_960.py
```

核心参数：

```python
model = YOLO('/home/hzh/container_yolo/yolo11m.pt')

model.train(
    data='/home/hzh/container_yolo/merged_container_number_yolo_b_rf/data.yaml',
    epochs=100,
    imgsz=960,
    batch=32,
    device='0,1,2,3',
    workers=16,
    project='/home/hzh/container_yolo/runs',
    name='region_b_rf_yolo11m_960_100e',
    exist_ok=True,
    patience=30,
)
```

启动命令：

```bash
python3 scripts/train_b_rf_960.py
```

服务器上使用 tmux 运行：

```bash
tmux new-session -d -s container_b_rf_960 \
  'cd /home/hzh/container_yolo && python3 train_b_rf_960.py 2>&1 | tee /home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e_train.log'
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

修改 `scripts/train_bonly_1280.py` 中的数据路径：

```python
data='/absolute/path/to/your/data.yaml'
```

如果只有一张 GPU，修改：

```python
device='0'
batch=4 或 batch=8
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
python3 scripts/train_b_rf_960.py
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
