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
nc: 1
names: ['container_number_region']
```

注意：B 当前没有独立 test 集，不应把 `valid/images` 当成正式 test。`valid/images` 只能用于训练过程验证和人工抽查；最终业务评估需要另建独立测试集或负例挑战集。

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

- 只有人工筛选确认没有清晰集装箱编号的图片，才可作为空标签负样本加入。
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

### 5.2 A 数据集试转、审核后合并

A 不能直接全量合并，也不能先全量合成最小外接框后再抽查。正确顺序是先做小规模试转和可视化审核，再决定是否使用简单外接框或空间聚类。

试转流程：

1. 从 A 的 `train` 和 `valid` 中随机抽取约 100 张有标签图片。
2. 读取每张图片对应的 YOLO label。
3. 将每个框从 `x_center y_center width height` 转为 `x1 y1 x2 y2`。
4. 先按“同一编号区域”假设计算最小外接矩形：
   - `x1_min = min(all_x1)`
   - `y1_min = min(all_y1)`
   - `x2_max = max(all_x2)`
   - `y2_max = max(all_y2)`
5. 将外接矩形转回 YOLO 格式。
6. 类别统一写成 `0`。
7. 输出试转可视化图片，人工检查合成框是否覆盖完整集装箱编号区域。

试转输出标签示例：

```text
0 x_center y_center width height
```

判定规则：

- 如果 100 张试转样本中，大部分图片都只有一个紧凑编号区域，且外接框没有明显包含大面积无关背景，可以使用简单最小外接框转换。
- 如果多簇编号区域比例超过 5%，或者外接框明显过大，应改为按空间聚类生成多个编号区域框。
- 如果某些图片的原始小框不是集装箱编号区域，应剔除这些图片，不加入正样本。

空间聚类规则建议：

1. 将同一图片内的原始小框按中心点坐标排序。
2. 根据横向和纵向间距把相邻小框聚为同一组。
3. 每组分别计算最小外接矩形。
4. 每组输出一个类别为 `0` 的编号区域框。
5. 聚类结果必须再次可视化抽查。

加入策略：

- 只有通过试转审核的 A 样本才加入合并集。
- A 的 `train` 转换后加入合并集 `train`。
- A 的 `valid` 转换后加入合并集 `val`。
- 如果 A 转换质量不稳定，首版只使用审核通过的部分，不强行全量加入。

命名前缀建议：

```text
a_train_原文件名.jpg
a_val_原文件名.jpg
```

### 5.3 C 数据集筛选后作为负样本

规则：

- 不使用 C 的原始检测标签作为正样本。
- C 的 `train` 和 `validation` 主要是集装箱、车辆、船等物体检测标注，不是干净的编号区域负样本。
- C 的 `negative` 也不能默认视为干净负样本，因为其中可能存在清晰集装箱编号或伪标注。
- 首版默认不加入 C；只有人工筛选确认“没有清晰集装箱编号区域”的图片，才可以作为空标签负样本加入。
- 如果图片中有清晰集装箱编号，但没有编号区域框，不能作为空标签负样本；要么剔除，要么手工补标后作为正样本。

筛选标准：

可以作为负样本：

- 没有集装箱编号的港口、道路、车辆、设备、建筑背景。
- 有车牌、车辆编号、设备编号、建筑编号，但没有清晰集装箱编号。
- 有集装箱但箱号不可见、严重模糊、过小到无法作为有效目标。

不能作为空标签负样本：

- 有清晰集装箱箱号但没有编号区域标注。
- 箱号被部分遮挡但仍可定位编号区域。
- 一张图中既有非目标编号，也有清晰集装箱编号。

加入方式：

- 为每张通过筛选的图片创建同名空 `.txt` 标签文件。
- 主要加入 `images/train` 和 `labels/train`。
- 少量加入 `images/val`，用于观察负样本误检。
- 如果人工筛选成本太高，第一版合并训练禁用 C，只训练 B+A。

命名前缀建议：

```text
c_neg_原文件名.jpg
c_train_bg_原文件名.jpg
c_val_bg_原文件名.jpg
```

建议加入数量：

```text
第一版: 0 张或仅加入人工确认干净的少量负样本
后续版: 根据误检类型逐步补充 100 到 200 张负样本
```

注意：

C 的价值是降低对非集装箱编号的误检，不是增加正样本。宁可少加，也不能把带清晰箱号的图片当空标签加入，否则会教模型不要检测真正目标。

## 6. 第一版合并比例

推荐按阶段合并，不直接一次性上 A+B+C：

```text
阶段 1: B-only，建立干净基线
阶段 2: B + 审核通过的 A，判断 A 是否提升召回或泛化
阶段 3: B + 审核通过的 A + 筛选后的 C 负样本，判断负样本是否降低非目标编号误检
```

第一版建议：

```text
B: 全部加入
A: 先 100 张试转可视化，通过后再批量转换；质量不稳定时只加入审核通过部分
C: 默认不加入；只有人工确认无清晰箱号后，才作为空标签负样本加入
```

比例控制原则：

- 不只按图片数量决定比例，要统计正样本实例数、负样本数量和来源域分布。
- C 负样本不宜一次加入过多，避免空标签噪声压制真正目标。
- 如果 B+A 的召回下降，优先回查 A 的转换质量。
- 如果 B+A+C 的召回下降，优先回查 C 是否含有未标注的清晰集装箱编号。

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

### 7.2 训练 B+A 模型

目的：单独验证 A 转换数据是否带来收益，避免把 A 和 C 的影响混在一起。

训练脚本：

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="/home/hzh/container_yolo/merged_container_number_yolo_ba/data.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device="0,1,2,3",
    workers=16,
    project="/home/hzh/container_yolo/runs",
    name="region_ba_yolo11m_100e",
    exist_ok=True,
    patience=30,
)
```

### 7.3 训练 B+A+C 模型

前提：C 中已有人工筛选通过的干净负样本。如果 C 未完成筛选，则跳过本阶段，不要强行加入空标签。

训练脚本：

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="/home/hzh/container_yolo/merged_container_number_yolo_bac/data.yaml",
    epochs=100,
    imgsz=640,
    batch=64,
    device="0,1,2,3",
    workers=16,
    project="/home/hzh/container_yolo/runs",
    name="region_bac_yolo11m_100e",
    exist_ok=True,
    patience=30,
)
```

### 7.4 对比指标

按阶段对比三个模型：

```text
B-only -> B+A -> B+A+C
```

- Precision
- Recall
- mAP50
- mAP50-95
- 误检数量
- 漏检数量
- 非集装箱编号误检情况
- 负例挑战集上的误检数量

重点不是只看 mAP，还要看业务目标：

```text
是否只检测集装箱编号，是否误检其他物体编号。
```

## 8. 验证计划

训练完成后分别检查三个阶段模型的训练输出：

```text
runs/region_b_yolo11m_100e/results.png
runs/region_b_yolo11m_100e/val_batch*_pred.jpg
runs/region_b_yolo11m_100e/confusion_matrix.png

runs/region_ba_yolo11m_100e/results.png
runs/region_ba_yolo11m_100e/val_batch*_pred.jpg
runs/region_ba_yolo11m_100e/confusion_matrix.png

runs/region_bac_yolo11m_100e/results.png
runs/region_bac_yolo11m_100e/val_batch*_pred.jpg
runs/region_bac_yolo11m_100e/confusion_matrix.png
```

注意：

- B 当前没有独立 test，`valid/images` 不能当作最终测试集。
- `valid/images` 可以用于训练过程验证和人工可视化抽查。
- 最终业务验证应额外建立独立负例挑战集。

负例挑战集建议：

```text
/home/hzh/container_yolo/container_number_negative_challenge/images
/home/hzh/container_yolo/container_number_negative_challenge/labels
```

负例挑战集应包含：

- 车牌编号。
- 车辆车身编号。
- 港口设备编号。
- 建筑编号。
- 路牌编号。
- 非集装箱文字和数字。
- 有集装箱但箱号不可见或不可读的图片。

负例挑战集标签应为空 `.txt` 文件，用于专门统计非目标编号误检。

推理验证建议：

```python
from ultralytics import YOLO

model = YOLO("/home/hzh/container_yolo/runs/region_bac_yolo11m_100e/weights/best.pt")

model.predict(
    source="/home/hzh/container_yolo/container_number_negative_challenge/images",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    save_crop=True,
    project="/home/hzh/container_yolo/runs",
    name="region_negative_challenge_predict",
    exist_ok=True,
)
```

如果没有训练 B+A+C，则用 B+A 的 `best.pt` 做同样验证。

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
2. 再完成 B+A 训练，单独判断 A 转换数据是否有效。
3. 人工筛选 C 或另建负例挑战集。
4. 如果 C 有干净负样本，再完成 B+A+C 训练。
5. 比较三个阶段模型。
6. 根据误检和漏检决定是否补数据。

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

- 先做 100 张试转可视化，不先全量合并。
- 统计多簇编号区域比例和明显过大框比例。
- 如果多簇率超过 5%，或外接框明显包含多个独立编号区域，必须先按空间距离聚类后生成多个区域框。
- 只把审核通过的 A 样本加入合并集。

### 10.2 C 数据集负样本风险

C 中可能包含清晰集装箱编号，但没有编号区域标注。

如果作为空标签加入，会教模型不要检测这些编号，造成漏检。

应对方式：

- 首版默认禁用 C，除非完成逐张人工筛选。
- 只选没有清晰集装箱编号的图片作为空标签负样本。
- 有清晰集装箱编号的图片不能空标；要么剔除，要么手工补标后再作为正样本。
- 如果加入 C 后召回下降，优先检查 C 是否包含未标注箱号。

### 10.3 课程合并要求与任务正确性的冲突

课程要求合并数据集，但模型目标必须一致。

应对方式：

- 不是简单拼接，而是统一标注语义后合并。
- B 直接加入，作为主正样本和基线。
- A 先试转和审核，通过后转换为编号区域框。
- C 不默认加入，只有筛选为干净负样本后才加入。
- 用 B-only、B+A、B+A+C 分阶段实验满足课程合并要求，同时保留问题定位能力。

## 11. 建议执行顺序

1. 从 A 中抽取约 100 张做试转和可视化审核。
2. 根据 A 试转结果决定使用简单外接框还是空间聚类。
3. 生成 B-only 数据配置，保留干净基线。
4. 生成 B+A 合并数据集。
5. 随机抽查 B+A 标签可视化。
6. 训练 B-only 正式模型。
7. 训练 B+A 正式模型。
8. 建立独立负例挑战集，专测非集装箱编号误检。
9. 人工筛选 C 中没有清晰箱号的图片；如果筛选成本过高，首版跳过 C。
10. 如果 C 筛选合格，生成 B+A+C 数据集并训练正式模型。
11. 对比 B-only、B+A、B+A+C 的指标和误检类型。
12. 根据误检和漏检决定是否补充负样本或特殊场景数据。
13. 用最佳检测模型裁剪编号区域。
14. 接 OCR 模块识别编号文本。

## 12. 给审核 agent 的核心问题

请重点审核以下问题：

1. A 数据集是否适合通过最小外接框转换为编号区域检测数据？
2. 如果 A 中一张图存在多个独立编号区域，是否需要空间聚类而不是全图合一个框？
3. C 数据集作为空标签负样本是否会引入漏检风险？
4. B-only 与 merged 两阶段训练对比是否足够证明合并有效？
5. 是否需要先人工抽查并剔除非集装箱编号样本？
6. 对“只检测集装箱编号，其他编号不要”的目标，是否还需要更多负样本？
