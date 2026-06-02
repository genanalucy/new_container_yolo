from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="集装箱编号区域检测")
    parser.add_argument("source", help="图片、视频或图片文件夹路径")
    parser.add_argument("--model", default=str(repo_root / "models" / "best_bonly_1280.pt"), help="模型权重路径")
    parser.add_argument("--output", default=str(repo_root / "outputs" / "predict_results"), help="预测结果输出目录")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理图片尺寸")
    parser.add_argument("--conf", type=float, default=0.05, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--device", default="cpu", help="设备，例如 cpu、0、mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"输入不存在: {source_path}")

    try:
        YOLO = importlib.import_module("ultralytics").YOLO
    except ImportError as exc:
        raise SystemExit("当前 Python 环境未安装 ultralytics，请先运行: python3 -m pip install ultralytics") from exc

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(output_dir),
        name="container_number_region",
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
    )

    for index, result in enumerate(results, start=1):
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        print(f"image_{index}: detections={count}")
    print(f"结果目录: {output_dir / 'container_number_region'}")


if __name__ == "__main__":
    main()
