"""Train a YOLOv8 or YOLO26 detector on the AUV YOLO dataset."""

import argparse
import os
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(
    "C:/Users/aiden/Downloads/AUV YOLO.v4-auvc_yolo26_dataset.yolo26"
    "/AUV YOLO.v4-auvc_yolo26_dataset.yolo26"
)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_data_yaml(dataset_dir: Path, out_path: Path) -> Path:
    """Rewrite a Roboflow data.yaml with absolute split paths.

    Roboflow exports use paths like ``../train/images`` that resolve relative to
    the global datasets dir rather than the export folder, so training silently
    fails to find images unless the paths are pinned.
    """
    source = dataset_dir / "data.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"No data.yaml found in {dataset_dir}")

    with source.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    resolved = {"path": str(dataset_dir)}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        images = dataset_dir / folder / "images"
        if images.is_dir():
            resolved[split] = str(images)
    resolved["nc"] = config["nc"]
    resolved["names"] = config["names"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    return out_path


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink an image into the merged dataset, falling back to a copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_merged_dataset(dataset_dir: Path, merge: list[str], merged_name: str,
                         out_dir: Path) -> Path:
    """Write a copy of the dataset with the named classes collapsed into one.

    Ultralytics can only collapse *every* class (``single_cls``), so merging a
    subset means remapping the label files. Images are hardlinked rather than
    copied, so the duplicate dataset costs almost no disk space.
    """
    names = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))["names"]
    by_lower = {n.lower(): i for i, n in enumerate(names)}

    unknown = [m for m in merge if m.lower() not in by_lower]
    if unknown:
        raise SystemExit(f"Unknown class(es) {unknown}. Dataset defines: {names}")

    merge_idx = {by_lower[m.lower()] for m in merge}
    if len(merge_idx) < 2:
        raise SystemExit("--merge needs at least two distinct classes")

    kept = [n for i, n in enumerate(names) if i not in merge_idx]
    new_names = kept + [merged_name]

    remap, next_idx = {}, 0
    for i in range(len(names)):
        if i in merge_idx:
            remap[i] = len(kept)
        else:
            remap[i] = next_idx
            next_idx += 1

    if out_dir.exists():
        shutil.rmtree(out_dir)

    resolved = {"path": str(out_dir)}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        src_images = dataset_dir / folder / "images"
        if not src_images.is_dir():
            continue
        dst_images = out_dir / folder / "images"
        dst_labels = out_dir / folder / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img in src_images.iterdir():
            link_or_copy(img, dst_images / img.name)
            src_label = dataset_dir / folder / "labels" / f"{img.stem}.txt"
            if not src_label.is_file():
                continue
            rows = []
            for line in src_label.read_text().strip().splitlines():
                parts = line.split()
                if parts:
                    parts[0] = str(remap[int(parts[0])])
                    rows.append(" ".join(parts))
            (dst_labels / f"{img.stem}.txt").write_text("\n".join(rows) + "\n" if rows else "")

        resolved[split] = str(dst_images)

    resolved["nc"] = len(new_names)
    resolved["names"] = new_names
    out_yaml = out_dir / "data.yaml"
    with out_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)

    merged_from = ", ".join(names[i] for i in sorted(merge_idx))
    print(f"merged [{merged_from}] -> '{merged_name}'; classes now {new_names}")
    return out_yaml


def set_channels(data_yaml: Path, channels: int) -> None:
    """Pin the model's input channel count via the dataset yaml.

    Ultralytics reads this as ``DetectionModel(ch=data["channels"])``, so
    ``channels: 1`` builds a genuinely single-channel network rather than
    feeding it a grey image replicated across three channels.
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["channels"] = channels
    with data_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Folder containing data.yaml plus train/valid/test splits")
    parser.add_argument("--arch", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolo26n", "yolo26s"],
                        help="Architecture to fine-tune. yolo26n is smaller (6.1 vs 8.2 GFLOPs), "
                             "DFL-free (reg_max=1) and NMS-free (end2end), which suits int8 "
                             "quantisation and Raspberry Pi deployment better than yolov8n")
    parser.add_argument("--model", default=None,
                        help="Explicit weights or .yaml, overriding --arch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256,
                        help="Match the source images (256px here); upscaling adds no information")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size; -1 auto-scales to available GPU memory")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or a CUDA index like 0")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=25,
                        help="Stop after this many epochs without improvement")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of the train split to use, for quick smoke tests")
    parser.add_argument("--single-cls", action="store_true",
                        help="Collapse every class into one; maximises detection recall when "
                             "you only need the target's position, not its aspect")
    parser.add_argument("--merge", nargs="+", metavar="CLASS",
                        help="Collapse just these classes into one, e.g. --merge front Side back")
    parser.add_argument("--merged-name", default="auv",
                        help="Label for the class created by --merge")
    parser.add_argument("--save-as", type=Path,
                        help="Also copy the best weights here, e.g. --save-as auv_merged.pt")
    parser.add_argument("--p2", action="store_true",
                        help="Add a stride-4 detection head for small/distant targets. At 256px "
                             "this gives a 64x64 finest grid instead of 32x32, at ~55%% more "
                             "FLOPs. Only the backbone transfers from pretrained weights")
    parser.add_argument("--grayscale", action="store_true",
                        help="Train a single-channel model. Underwater colour is dominated by "
                             "depth-dependent attenuation, so dropping it can generalise better "
                             "across water conditions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--name", default=None, help="Run name; defaults to <arch>_auv")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run of the same name")
    parser.add_argument("--export", choices=["none", "onnx", "torchscript"], default="onnx",
                        help="Format to export the best weights to after training")

    args = parser.parse_args()
    args.weights = args.model or f"{args.arch}.pt"
    args.name = args.name or f"{args.arch}_auv"
    return args


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    if args.merge:
        data_yaml = build_merged_dataset(
            args.dataset, args.merge, args.merged_name, REPO_ROOT / "datasets" / args.name
        )
    else:
        data_yaml = resolve_data_yaml(args.dataset, REPO_ROOT / "data.resolved.yaml")

    if args.grayscale:
        set_channels(data_yaml, 1)

    print(f"device: {device}")
    print(f"data:   {data_yaml}")
    print(f"model:  {args.weights}")
    print(f"channels: {1 if args.grayscale else 3}")

    if args.p2:
        # No pretrained P2 checkpoints are published, so build the architecture
        # from its yaml and transfer whatever the standard weights can supply.
        model = YOLO(f"{args.arch}-p2.yaml").load(args.weights)
    else:
        model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        single_cls=args.single_cls,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        resume=args.resume,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        plots=True,
    )

    metrics = model.val(
        split="test",
        device=device,
        single_cls=args.single_cls,
        project=str(args.project),
        name=f"{args.name}_test",
    )
    print(f"test mAP50-95: {metrics.box.map:.4f}")
    print(f"test mAP50:    {metrics.box.map50:.4f}")

    exported = None
    if args.export != "none":
        exported = Path(model.export(format=args.export, imgsz=args.imgsz))
        print(f"exported: {exported}")

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")

    if args.save_as:
        dest = args.save_as if args.save_as.is_absolute() else REPO_ROOT / args.save_as
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"saved as: {dest}")
        if exported is not None:
            dest_export = dest.with_suffix(exported.suffix)
            shutil.copy2(exported, dest_export)
            print(f"saved as: {dest_export}")


if __name__ == "__main__":
    main()
