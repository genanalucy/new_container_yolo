from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="训练 B + Roboflow 960 集装箱编号区域检测模型")
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO 预训练权重路径或模型名")
    parser.add_argument("--data", default=str(repo_root / "datasets" / "merged_container_number_yolo_b_rf" / "data.yaml"), help="YOLO data.yaml 路径")
    parser.add_argument("--project", default=str(repo_root / "runs"), help="训练输出目录")
    parser.add_argument("--name", default="region_b_rf_yolo11m_960_100e", help="实验名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=960, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--device", default="0", help="训练设备，例如 0 或 0,1,2,3")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=30, help="early stopping 轮数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    project_path = Path(args.project).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"数据配置不存在: {data_path}")

    try:
        YOLO = importlib.import_module("ultralytics").YOLO
    except ImportError as exc:
        raise SystemExit("当前 Python 环境未安装 ultralytics，请先运行: python3 -m pip install -r requirements.txt") from exc

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
