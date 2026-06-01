# 集装箱编号检测数据集合并计划

## 1. 项目目标

本项目目标是训练 YOLO 模型，只检测集装箱箱体上的编号区域。

非目标包括：

- 车牌编号
- 货车车身编号
- 港口设备编号
- 建筑编号
- 路牌编号
- 其他非集装箱物体上的数字或文字

最终类别统一为：

```yaml
nc: 1
names: ['container_number_region']
```

## 2. 已调查的数据集

远端数据根目录：

```text
/home/hzh/container_yolo
```

### 2.1 数据集 A

路径：

```text
/home/hzh/container_yolo/container_v1i_yolov11_a
```

结构：

```text
train/images: 5784
train/labels: 5784
valid/images: 1979
valid/labels: 1979
test/images: 0
test/labels: 0
```

原始配置：

```yaml
nc: 4
names: ['0', '1', '2', '3']
```

样例标签：

```text
2 0.346875 0.4112734864300626 0.0125 0.029749478079331943
3 0.3484375 0.3298538622129436 0.016015625 0.19258872651356992
0 0.3484375 0.2755741127348643 0.03359375 0.3094989561586639
1 0.3515625 0.17536534446764093 0.021875 0.0965553235908142
```

判断：

A 数据集不是完整编号区域一类检测数据，而是把编号区域拆成多个局部结构或子区域。它不能直接和 B 数据集合并，否则会造成标注语义混乱。

可用方式：

- 将同一张图片中的多个局部框合并成一个最小外接大框。
- 忽略原始类别 `0/1/2/3`。
- 输出统一类别 `0: container_number_region`。
- 转换后作为编号区域正样本加入合并集。

### 2.2 数据集 B

路径：

```text
/home/hzh/container_yolo/container_v1i_yolov11_b
```

结构：

```text
train/images: 5838
train/labels: 5838
valid/images: 451
valid/labels: 451
test/images: 0
test/labels: 0
```

修正后配置：

```yaml
path: /home/hzh/container_yolo/container_v1i_yolov11_b
train: train/images
val: valid/images
test: valid/images
nc: 1
names: ['container_number_region']
```

样例标签：

```text
0 0.46484375 0.296875 0.33125 0.5453125
```

判断：

B 数据集与本项目目标最一致，是主数据集。它已经是 YOLO 检测格式，且类别语义可以统一为 `container_number_region`。

可用方式：

- 直接合并。
- `train` 加入合并训练集。
- `valid` 加入合并验证集。
- 类别保持为 `0`。

### 2.3 数据集 C

路径：

```text
/home/hzh/container_yolo/container_detection_master/container_detection-master
```

结构：

```text
datasets/train: 226 images
datasets/validation: 27 images
datasets/negative: 24 images
```

类别文件：

```yaml
- container
- truck
- ship
- boat
- car
- bus
- bench
```

判断：

C 数据集是老的 ChainerCV / SSD 项目，目标是检测集装箱、车辆、船、公交、长椅等物体，不是检测集装箱编号区域。

因此，C 不能作为编号区域正样本直接加入训练。

可用方式：

- 作为背景负样本加入。
- 对应 YOLO 标签文件设为空文件。
- 用于减少模型对非集装箱编号、车辆、建筑、港口背景等区域的误检。

## 3. 三个数据集的共性与差异

### 3.1 共性

三个数据集的共同点：

- 都是图像检测相关数据。
- 都和集装箱或港口物流场景有关。
- 都可以间接服务于集装箱编号识别任务。
- 最终都可以整理到 YOLO 训练目录中。

### 3.2 差异

关键差异是标注语义不同：

| 数据集 | 原始任务 | 标注格式 | 是否可直接作为编号区域正样本 |
| --- | --- | --- | --- |
| A | 编号局部结构或子区域检测 | YOLO, 4 类 | 否，需要合并框转换 |
| B | 编号区域检测 | YOLO, 1 类 | 是 |
| C | 集装箱/车辆/船等物体检测 | JSON/SSD 旧格式 | 否，只建议做负样本 |

## 4. 合并后目标数据集

建议创建新目录：

```text
/home/hzh/container_yolo/merged_container_number_yolo
```

目录结构：

```text
merged_container_number_yolo/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

配置文件：

```yaml
path: /home/hzh/container_yolo/merged_container_number_yolo
train: images/train
val: images/val
nc: 1
names: ['container_number_region']
```

## 5. 合并规则

### 5.1 B 数据集直接合并

规则：

- 拷贝 B 的 `train/images` 到合并集 `images/train`。
- 拷贝 B 的 `train/labels` 到合并集 `labels/train`。
- 拷贝 B 的 `valid/images` 到合并集 `images/val`。
- 拷贝 B 的 `valid/labels` 到合并集 `labels/val`。
- 标签类别保持为 `0`。

注意：

为了避免不同数据集文件名冲突，建议统一加前缀：

```text
b_train_原文件名.jpg
b_val_原文件名.jpg
```

对应标签文件同名改写。

### 5.2 A 数据集转换后合并

规则：

对 A 中每张图片：

1. 读取该图片对应的 YOLO label。
2. 将每个框从 `x_center y_center width height` 转为 `x1 y1 x2 y2`。
3. 计算所有框的最小外接矩形：
   - `x1_min = min(all_x1)`
   - `y1_min = min(all_y1)`
   - `x2_max = max(all_x2)`
   - `y2_max = max(all_y2)`
4. 将外接矩形转回 YOLO 格式。
5. 类别统一写成 `0`。
6. 输出一个标签文件，通常每张图一行：

```text
0 x_center y_center width height
```

加入策略：

- A 的 `train` 转换后加入合并集 `train`。
- A 的 `valid` 转换后加入合并集 `val`。

质量控制：

- 转换后必须抽查可视化 50 到 100 张。
- 如果发现某些图片外接框不是完整集装箱编号区域，应剔除或单独修正。
- 如果一张图中存在多个独立编号区域，简单外接框可能会过大，需要改成按空间聚类分成多个编号区域框。

命名前缀建议：

```text
a_train_原文件名.jpg
a_val_原文件名.jpg
```

### 5.3 C 数据集作为负样本合并

规则：

- 不使用 C 的原始检测标签作为正样本。
- 从 C 中选择背景或非编号图片加入训练集。
- 为每张加入的图片创建同名空 `.txt` 标签文件。

建议加入数量：

```text
negative: 全部 24 张
train/validation: 抽取 100 到 200 张
```

加入位置：

- 主要加入 `images/train` 和 `labels/train`。
- 少量加入 `images/val`，用于观察负样本误检。

命名前缀建议：

```text
c_neg_原文件名.jpg
c_train_bg_原文件名.jpg
c_val_bg_原文件名.jpg
```

注意：

C 中如果存在清晰集装箱编号但没有编号区域标注，把它当空标签会造成训练噪声。加入前应抽查，优先选择没有清晰编号的背景图。

## 6. 第一版合并比例

推荐第一版：

```text
B: 全部加入
A: 转换后全部加入，但先抽查质量
C: 加入 100 到 200 张负样本
```

如果 A 转换效果不稳定，则改成：

```text
B: 全部加入
A: 转换后只加入抽查通过的部分
C: 加入 100 到 200 张负样本
```

## 7. 训练计划

### 7.1 先保留 B-only 基线

目的：有一个干净基线，便于判断合并数据是否真的提升。

训练脚本：

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="/home/hzh/container_yolo/container_v1i_yolov11_b/data_fixed.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device="0,1,2,3",
    workers=16,
    project="/home/hzh/container_yolo/runs",
    name="region_b_yolo11m_100e",
    exist_ok=True,
    patience=30,
)
```

### 7.2 训练合并数据集

训练脚本：

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="/home/hzh/container_yolo/merged_container_number_yolo/data.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device="0,1,2,3",
    workers=16,
    project="/home/hzh/container_yolo/runs",
    name="region_merged_yolo11m_100e",
    exist_ok=True,
    patience=30,
)
```

### 7.3 对比指标

对比 B-only 与 merged 模型：

- Precision
- Recall
- mAP50
- mAP50-95
- 误检数量
- 漏检数量
- 非集装箱编号误检情况

重点不是只看 mAP，还要看业务目标：

```text
是否只检测集装箱编号，是否误检其他物体编号。
```

## 8. 验证计划

训练完成后检查：

```text
runs/region_merged_yolo11m_100e/results.png
runs/region_merged_yolo11m_100e/val_batch*_pred.jpg
runs/region_merged_yolo11m_100e/confusion_matrix.png
```

推理验证建议：

```python
from ultralytics import YOLO

model = YOLO("/home/hzh/container_yolo/runs/region_merged_yolo11m_100e/weights/best.pt")

model.predict(
    source="/home/hzh/container_yolo/merged_container_number_yolo/images/val",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    save_crop=True,
    project="/home/hzh/container_yolo/runs",
    name="region_merged_predict_crops",
    exist_ok=True,
)
```

人工重点检查：

- 是否漏检集装箱编号。
- 是否把车牌框出来。
- 是否把车辆编号框出来。
- 是否把建筑编号框出来。
- 是否把港口设备编号框出来。
- 是否把非集装箱文字框出来。

## 9. 是否需要补充数据

当前不建议先盲目补充新数据集。

建议流程：

1. 先完成 B-only 基线。
2. 再完成 A+B+C 合并训练。
3. 比较两个模型。
4. 根据误检和漏检决定是否补数据。

如果需要补充，优先补：

- 非集装箱编号负样本。
- 车牌负样本。
- 车辆编号负样本。
- 港口设备编号负样本。
- 夜间集装箱编号。
- 雨天、反光、模糊编号。
- 远距离小编号。
- 竖排编号。
- 污损编号。
- 多集装箱同框。
- 局部遮挡编号。

## 10. 风险点

### 10.1 A 数据集转换风险

A 的多个小框如果属于同一个编号区域，外接框转换合理。

但如果一张图中有多个独立编号区域，简单合成一个大框会产生过大的错误框。

应对方式：

- 先抽查。
- 必要时按空间距离聚类后生成多个区域框。

### 10.2 C 数据集负样本风险

C 中可能包含清晰集装箱编号，但没有编号区域标注。

如果作为空标签加入，会教模型不要检测这些编号，造成漏检。

应对方式：

- 只选没有清晰编号的图片作为负样本。
- 或者手工补标后再作为正样本。

### 10.3 课程合并要求与任务正确性的冲突

课程要求合并数据集，但模型目标必须一致。

应对方式：

- 不是简单拼接，而是统一标注语义后合并。
- A 转换为区域框。
- B 直接加入。
- C 作为负样本加入。

## 11. 建议执行顺序

1. 可视化抽查 A 的转换可行性。
2. 编写合并脚本。
3. 生成 `merged_container_number_yolo`。
4. 随机抽查合并后的标签可视化。
5. 训练 B-only 正式模型。
6. 训练 merged 正式模型。
7. 对比两个模型指标和误检类型。
8. 决定是否补充负样本或特殊场景数据。
9. 用最佳检测模型裁剪编号区域。
10. 接 OCR 模块识别编号文本。

## 12. 给审核 agent 的核心问题

请重点审核以下问题：

1. A 数据集是否适合通过最小外接框转换为编号区域检测数据？
2. 如果 A 中一张图存在多个独立编号区域，是否需要空间聚类而不是全图合一个框？
3. C 数据集作为空标签负样本是否会引入漏检风险？
4. B-only 与 merged 两阶段训练对比是否足够证明合并有效？
5. 是否需要先人工抽查并剔除非集装箱编号样本？
6. 对“只检测集装箱编号，其他编号不要”的目标，是否还需要更多负样本？
