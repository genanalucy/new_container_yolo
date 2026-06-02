# 公开集装箱编号数据集补充与模型提升计划

状态：pending approval

## 1. 背景与当前问题

当前目标是训练 YOLO 模型，只检测集装箱箱体上的编号区域，不检测车牌、车辆编号、港口设备编号、建筑编号、路牌编号或其他非集装箱文字数字。

当前已完成两组训练：

| 模型 | 数据 | Precision | Recall | mAP50 | mAP50-95 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B-only | 原 B 数据集 | 0.99786 | 0.99768 | 0.99500 | 0.82042 | 当前最优 |
| B+A | B + A 聚类转换 | 0.96794 | 0.96885 | 0.97302 | 0.80338 | 合并 A 后下降 |

当前最佳权重：

```text
/home/hzh/container_yolo/runs/region_b_yolo11m_100e/weights/best.pt
```

实际测试三张港口图片时，使用 `imgsz=1280, conf=0.05` 才能检出部分目标：

| 图 | 检出情况 | 结论 |
| --- | --- | --- |
| pred_01 | conf 约 0.08/0.11，位置不准 | 误检/弱检 |
| pred_02 | conf 约 0.57，位置正确 | 可用 |
| pred_03 | conf 约 0.55，位置正确 | 可用 |

本地预测结果图：

```text
/Users/bb/myfiles/AI_detect/tmp/pred_01.jpg
/Users/bb/myfiles/AI_detect/tmp/pred_02.jpg
/Users/bb/myfiles/AI_detect/tmp/pred_03.jpg
```

核心问题：

1. 远距离小编号召回不足。
2. 复杂港口背景下置信度偏低。
3. A 数据集合并后指标下降，说明 A 转换框存在语义噪声或分布干扰。
4. 缺少专门的非集装箱编号负样本验证集。
5. 目前验证指标高，但真实场景测试图表现不够稳定，说明验证集与业务场景分布不完全一致。

## 2. 可参考的公开数据集

### 2.1 Roboflow Container Number Detection v2

链接：

```text
https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/2
```

已知信息：

- 任务类型：集装箱编号检测。
- 数据量：约 5,043 张图片。
- 划分：约 4,449 train / 231 valid / 363 test。
- 支持导出 YOLOv8 / YOLOv9 / YOLOv11 等格式。
- 与本项目最匹配，优先级最高。

使用前必须确认：

1. 标注是否是完整集装箱编号区域。
2. 是否存在单字符标注或分段标注。
3. 是否误标车牌、车辆编号、设备编号。
4. 类别是否能统一到 `container_number_region`。
5. 许可协议是否允许课程项目或后续使用。

初步用途：

- 作为 B-only 之后的主要补充正样本。
- 用来增强远距离、小编号、多集装箱同框场景。

### 2.2 Roboflow 其它 container-number 数据集

入口：

```text
https://universe.roboflow.com/search?q=container+number
https://universe.roboflow.com/search?q=class%3Acontainer+number
```

搜索结果中有多个候选：

- Container Number：约 334 张。
- container-number：约 299 张。
- Container number：约 3.43k 张。
- container-id-number：约 1.11k 张。
- container-number-detection：约 301 张。

使用策略：

- 不一次性全部合并。
- 每个数据集先抽查 50 到 100 张。
- 只保留标注语义一致的数据。
- 小数据集可作为特殊场景补充，不作为主训练集。

### 2.3 GitHub ContainerNumber-OCR

链接：

```text
https://github.com/lbf4616/ContainerNumber-OCR
```

数据集链接：

```text
https://drive.google.com/drive/folders/13LpHEeFExmDJnw_U9peqLR-8uAAUMEzi?usp=sharing
```

特点：

- 是检测 + OCR 识别项目。
- README 中提供数据集和预训练模型。
- 可能不是 YOLO 格式，需要转换。
- 更适合后续 OCR 模块参考。

使用前必须确认：

1. 是否有编号区域检测框。
2. 是否有字符级标注。
3. 图像分辨率和实际港口场景是否相似。
4. 是否能转换成 YOLO 单类检测格式。

初步用途：

- 如果有区域框：可作为正样本补充。
- 如果只有 OCR 文本或字符级标注：后续接 OCR 模块使用，不直接合并到检测模型。

### 2.4 HyperAI / OpenBayes ContainerNumber-OCR Dataset

链接：

```text
https://hyper.ai/cn/news/22766
```

已知信息：

- 资料显示包含约 3,003 张集装箱图片。
- 图片分辨率约 1920×1080。
- 偏 PaddleOCR 教程方向。

使用前必须确认：

1. 数据是否能直接下载。
2. 标注格式是检测框、字符框还是文本识别标签。
3. 是否能转换成 YOLO 检测数据。
4. 许可协议是否允许使用。

初步用途：

- 优先作为 OCR 阶段参考。
- 如果包含编号区域框，再纳入检测训练候选。

## 3. 数据补充原则

### 3.1 正样本原则

正样本只允许：

```text
集装箱箱体上的编号区域
```

不得作为正样本：

- 车牌编号。
- 货车车身编号。
- 港口设备编号。
- 建筑编号。
- 路牌编号。
- 集装箱品牌 Logo。
- 中国铁路、公司名、警示标识等非编号文字。
- 单个字符框，除非明确转换成完整编号区域框。

最终类别保持：

```yaml
nc: 1
names: ['container_number_region']
```

### 3.2 负样本原则

负样本用于降低误检，标签为空 `.txt`。

优先补充：

- 车牌清晰但无集装箱编号的图。
- 车辆编号清晰但无集装箱编号的图。
- 港口设备编号清晰但无集装箱编号的图。
- 建筑编号、路牌编号清晰但无集装箱编号的图。
- 集装箱不可见箱号的港口背景。
- 有集装箱但箱号太远、不可读、不可定位的图片。

不能作为空标签负样本：

- 有清晰集装箱编号但未标注的图。
- 一张图中同时有非目标编号和清晰集装箱编号的图。
- 箱号部分遮挡但仍可定位的图。

### 3.3 公开数据集验收门槛

每个新数据集进入训练前必须完成：

1. 随机抽查 50 到 100 张。
2. 统计类别数量和类别名。
3. 统计每张图片框数量。
4. 检查是否有单字符标注。
5. 检查是否存在非集装箱编号正样本。
6. 生成 contact sheet 可视化。
7. 写数据集审核摘要。

通过门槛：

- 正样本语义一致率 >= 95%。
- 明显错误标注比例 <= 3%。
- 非集装箱编号误标为正样本比例 <= 1%。
- 若错误集中在少数类别或目录，可筛除后使用。

## 4. 推荐实验路线

### 4.1 阶段 0：保留当前 B-only 基线

当前 B-only 是最优基线，不覆盖、不删除。

保留：

```text
/home/hzh/container_yolo/runs/region_b_yolo11m_100e
/home/hzh/container_yolo/runs/region_b_yolo11m_100e/weights/best.pt
```

### 4.2 阶段 1：高分辨率 B-only 重训

目的：验证小目标是否主要受 `imgsz` 限制。

训练建议：

```python
from ultralytics import YOLO

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

说明：

- 4 张 3090 可跑，但 1280 下 batch 需要降低。
- 如果显存不足，改为 `batch=8` 或 `imgsz=960`。
- 先不引入新数据，单独判断分辨率收益。

验收：

- 三张实测图中，图 2 和图 3 conf 明显提高。
- 图 1 不应继续误检楼梯。
- mAP50-95 不低于当前 B-only 太多。

### 4.3 阶段 2：Roboflow Container Number Detection 审核与合并

目录建议：

```text
/home/hzh/container_yolo/public_datasets/roboflow_container_number_detection_v2
/home/hzh/container_yolo/merged_container_number_yolo_b_rf
```

处理流程：

1. 下载 Roboflow YOLO 格式数据。
2. 统一类别为 `0 container_number_region`。
3. 抽查 100 张训练图和 50 张验证图。
4. 生成 contact sheet。
5. 写审核摘要。
6. 如果通过审核，生成 B + Roboflow 合并集。

合并策略：

```text
B: 全部加入
Roboflow train: 审核通过后加入 train
Roboflow valid/test: 可作为独立评估集，不建议全部混入 val
```

推荐保留 Roboflow 原 test 作为外部测试集：

```text
/home/hzh/container_yolo/eval_sets/roboflow_container_number_test
```

训练建议：

```python
from ultralytics import YOLO

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

### 4.4 阶段 3：建立负例挑战集

目录建议：

```text
/home/hzh/container_yolo/container_number_negative_challenge
  images/
  labels/
```

目标数量：

```text
第一版: 100 到 200 张
稳定版: 300 到 500 张
```

来源：

- 现有 C 数据集中人工筛选无清晰箱号图片。
- 用户真实港口图片中无箱号或箱号不可定位图片。
- Roboflow/公开数据集中非目标编号背景。
- 自行采集车牌、设备编号、建筑编号、货车编号场景。

标签：

- 每张图片一个同名空 `.txt`。

用途：

1. 不一定立即加入训练。
2. 先作为外部评估集，统计误检数量。
3. 确认误检类型后，再挑选一部分加入训练负样本。

验收：

- B-only 在该集合上的误检数作为基线。
- 新模型误检数应下降。
- 不允许通过牺牲真实箱号召回来换取低误检。

### 4.5 阶段 4：B + Roboflow + 负样本微调

前提：

- Roboflow 审核通过。
- 负例挑战集已建立。
- 已知道主要误检类型。

数据组成：

```text
B 正样本: 全部
Roboflow 正样本: 审核通过部分
负样本: 100 到 300 张干净空标签图片
```

训练建议：

```python
from ultralytics import YOLO

model = YOLO('/home/hzh/container_yolo/runs/region_b_rf_yolo11m_960_100e/weights/best.pt')

model.train(
    data='/home/hzh/container_yolo/merged_container_number_yolo_b_rf_neg/data.yaml',
    epochs=50,
    imgsz=960,
    batch=32,
    device='0,1,2,3',
    workers=16,
    project='/home/hzh/container_yolo/runs',
    name='region_b_rf_neg_yolo11m_960_50e',
    exist_ok=True,
    patience=20,
)
```

### 4.6 阶段 5：更大模型对比

如果 yolo11m 仍然小目标不足，再对比：

```text
yolo11l.pt
yolo11x.pt
```

建议顺序：

1. 先跑 yolo11l + imgsz=960。
2. 如果提升明显，再跑 yolo11x。
3. 不要一开始就上 x，先确认数据问题是否解决。

## 5. 推理策略优化

### 5.1 推荐默认推理参数

当前三张真实图显示，640 分辨率不够。

推荐默认：

```python
model.predict(
    source='...',
    imgsz=1280,
    conf=0.15,
    iou=0.5,
    device=0,
    save=True,
    save_crop=True,
)
```

### 5.2 小目标场景可使用切片推理

如果图片分辨率高、编号很小，可使用 SAHI 或自写切片：

```text
原图 -> 1024 或 1280 tile -> 重叠 20% -> YOLO 推理 -> NMS 合并
```

适用场景：

- 多集装箱远景。
- 监控摄像头全景。
- 车道口远距离箱号。

注意：

- 切片推理会增加耗时。
- 只建议用于离线验证或高价值图片。

## 6. 验证方案

### 6.1 必测集合

每个候选模型都必须在以下集合上验证：

1. B valid。
2. Roboflow test。
3. 用户真实 3 张测试图。
4. 负例挑战集。
5. 小目标挑战集。

### 6.2 指标

常规指标：

- Precision。
- Recall。
- mAP50。
- mAP50-95。

业务指标：

- 用户真实图检出数。
- 用户真实图平均 conf。
- 负例挑战集误检数。
- 小目标挑战集召回率。
- 是否误检车牌、车辆编号、设备编号。

### 6.3 模型选择规则

不能只看 mAP。

推荐选择规则：

1. B valid mAP50 不明显低于 B-only。
2. 小目标挑战集召回高于 B-only。
3. 负例挑战集误检数低于 B-only。
4. 用户真实图表现优于当前模型。
5. 若 mAP 略降但真实图提升明显，可以优先选择真实图更好的模型。

## 7. 文件与目录规划

服务器目录：

```text
/home/hzh/container_yolo/public_datasets/
/home/hzh/container_yolo/public_datasets/roboflow_container_number_detection_v2/
/home/hzh/container_yolo/public_datasets/container_number_ocr_github/
/home/hzh/container_yolo/public_datasets/hyperai_openbayes_container_ocr/

/home/hzh/container_yolo/eval_sets/
/home/hzh/container_yolo/eval_sets/roboflow_container_number_test/
/home/hzh/container_yolo/eval_sets/user_real_3_images/
/home/hzh/container_yolo/eval_sets/negative_challenge/
/home/hzh/container_yolo/eval_sets/small_target_challenge/

/home/hzh/container_yolo/merged_container_number_yolo_b_rf/
/home/hzh/container_yolo/merged_container_number_yolo_b_rf_neg/
```

本地计划与结果目录：

```text
/Users/bb/myfiles/AI_detect/public_container_number_dataset_improvement_plan.md
/Users/bb/myfiles/AI_detect/tmp/
```

## 8. 建议执行顺序

1. 保留当前 B-only 最佳模型，不覆盖。
2. 训练 B-only `imgsz=1280`，确认高分辨率收益。
3. 下载 Roboflow Container Number Detection v2。
4. 抽查 Roboflow 标注质量，生成审核摘要和 contact sheet。
5. 如果通过审核，生成 B + Roboflow 数据集。
6. 训练 B + Roboflow `imgsz=960` 或 `1280`。
7. 建立负例挑战集。
8. 用 B-only、B-only-1280、B+Roboflow 对负例挑战集和用户 3 图做推理对比。
9. 如果误检仍明显，加入负样本微调 50 epoch。
10. 如果小目标仍不足，再试 yolo11l。
11. 选择业务表现最好的模型作为检测模块。
12. 后续接 OCR 模块识别裁剪出来的编号文本。

## 9. 风险与应对

### 9.1 公开数据集标注语义不一致

风险：有些数据集可能标单字符、整箱、Logo 或非编号文字。

应对：

- 每个数据集先抽查。
- 不通过的数据不合并。
- 能转换的先转换，不能转换的只作为参考。

### 9.2 Roboflow 数据与当前业务场景分布不同

风险：指标上升但真实图不提升。

应对：

- 保留用户真实图和小目标挑战集作为固定评估集。
- 模型选择不只看 mAP。

### 9.3 负样本空标导致漏检

风险：负样本中有未标注箱号，会教模型不要检测目标。

应对：

- 负样本必须人工筛选。
- 有清晰箱号的图片不能空标。
- 负样本先少量加入，观察 Recall 是否下降。

### 9.4 高分辨率训练显存不足

风险：`imgsz=1280` batch 太大导致 OOM。

应对：

- 先 `batch=16`。
- OOM 改 `batch=8`。
- 仍不稳定改 `imgsz=960`。

## 10. 验收标准

第一阶段验收：

- `B-only-1280` 成功训练完成。
- 用户 3 张真实图至少 2 张正确检出。
- 图 1 误检减少或消失。
- 小目标 conf 相比当前模型提高。

第二阶段验收：

- Roboflow 数据审核通过并生成合并集。
- B + Roboflow 模型训练完成。
- 在 Roboflow test 和用户真实图上表现优于 B-only。

第三阶段验收：

- 负例挑战集建立完成。
- 新模型在负例挑战集误检低于 B-only。
- 不明显牺牲 B valid Recall。

最终验收：

- 选出一个业务最佳模型。
- 输出：

```text
best.pt
results.png
val_batch*_pred.jpg
用户真实图预测结果
负例挑战集误检统计
模型对比表
```

## 11. ADR

### Decision

优先采用“高分辨率 B-only 重训 + Roboflow Container Number Detection 审核合并 + 负例挑战集微调”的路线。

### Drivers

1. 当前 B-only 指标最高，但真实远景图片置信度偏低。
2. B+A 指标下降，说明继续依赖 A 转换数据风险较高。
3. 公开 Roboflow Container Number Detection 与当前任务语义最接近。
4. 业务目标要求减少非集装箱编号误检，必须建立负例挑战集。

### Alternatives considered

1. 继续清洗 A 数据集后训练。
   - 优点：已有数据，不需要下载公开数据。
   - 缺点：A 原始语义是局部结构，转换噪声已经导致 B+A 指标下降。

2. 直接换更大模型 yolo11x。
   - 优点：可能提升小目标召回。
   - 缺点：如果数据分布和标注问题不解决，收益有限。

3. 只降低 conf 阈值。
   - 优点：马上提升召回。
   - 缺点：误检会增加，图 1 已显示低 conf 会带来错误框。

### Why chosen

该路线先验证分辨率瓶颈，再引入语义更接近的公开数据，最后用负样本约束误检，能分别解决小目标、数据不足和复杂背景误检三个问题。

### Consequences

- 需要人工审核公开数据集，不能直接合并。
- 训练时间会增加。
- 需要维护额外的外部测试集和负例挑战集。

### Follow-ups

1. 下载 Roboflow Container Number Detection v2。
2. 建立数据审核脚本和 contact sheet。
3. 重训 B-only-1280。
4. 建立负例挑战集。
5. 对比所有候选模型。

## 12. 参考来源

- Roboflow Container Number Detection v2: https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/2
- Roboflow Container Number Detection project: https://universe.roboflow.com/pranw/container-number-detection-wcunq
- Roboflow container number search: https://universe.roboflow.com/search?q=container+number
- Roboflow class search: https://universe.roboflow.com/search?q=class%3Acontainer+number
- GitHub ContainerNumber-OCR: https://github.com/lbf4616/ContainerNumber-OCR
- ContainerNum_dataset Google Drive: https://drive.google.com/drive/folders/13LpHEeFExmDJnw_U9peqLR-8uAAUMEzi?usp=sharing
- HyperAI PaddleOCR container number tutorial: https://hyper.ai/cn/news/22766
- Container number OCR system article: https://linzichun.com/posts/robust-container-number-ocr-recognition-system-part2/
