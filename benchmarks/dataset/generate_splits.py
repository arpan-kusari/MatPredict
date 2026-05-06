import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def _is_valid_variant_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("_tmp"):
        return False
    return (path / "images").is_dir() and (path / "albedo").is_dir() and (path / "ORM").is_dir()


def _collect_sample_ids(dataset_root: str) -> List[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    sample_ids: List[str] = []
    for object_dir in sorted(root.iterdir()):
        if not object_dir.is_dir():
            continue
        object_id = object_dir.name

        for variant_dir in sorted(object_dir.iterdir()):
            if not _is_valid_variant_dir(variant_dir):
                continue
            var_id = variant_dir.name
            image_dir = variant_dir / "images"
            albedo_dir = variant_dir / "albedo"
            orm_dir = variant_dir / "ORM"

            for img_path in sorted(image_dir.glob("*.png")):
                frame_id = img_path.stem
                if (albedo_dir / f"{frame_id}.png").exists() and (orm_dir / f"{frame_id}.png").exists():
                    sample_ids.append(f"{object_id}/{var_id}/{frame_id}")

    if not sample_ids:
        raise RuntimeError("No valid samples found. Expected .../<object>/<varXXX>/{images,albedo,ORM}.")
    return sample_ids


def _group_sample_ids(sample_ids: List[str], strategy: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for sid in sample_ids:
        obj, var_id, _ = sid.split("/")
        if strategy == "object_disjoint":
            key = obj
        elif strategy == "variance_disjoint":
            key = var_id
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        groups[key].append(sid)
    return dict(groups)


def _balanced_assign(
    groups: Dict[str, List[str]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[str]]:
    split_names = ["train", "val", "test"]
    total = sum(len(v) for v in groups.values())
    targets = {
        "train": total * ratios[0],
        "val": total * ratios[1],
        "test": total * ratios[2],
    }

    counts = {"train": 0, "val": 0, "test": 0}
    splits = {"train": [], "val": [], "test": []}

    items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    for _, ids in items:
        best_split = None
        best_score = None
        for s in split_names:
            remaining = targets[s] - counts[s]
            score = (remaining, -counts[s])
            if best_score is None or score > best_score:
                best_score = score
                best_split = s
        splits[best_split].extend(ids)
        counts[best_split] += len(ids)

    for s in split_names:
        splits[s] = sorted(splits[s])
    return splits


def _write_split_yaml(
    output_path: Path,
    name: str,
    strategy: str,
    dataset_root: str,
    seed: int,
    ratios: Tuple[float, float, float],
    split_ids: Dict[str, List[str]],
) -> None:
    payload = {
        "name": name,
        "strategy": strategy,
        "seed": int(seed),
        "ratios": {
            "train": float(ratios[0]),
            "val": float(ratios[1]),
            "test": float(ratios[2]),
        },
        "dataset_root": str(dataset_root),
        "counts": {
            "train": len(split_ids["train"]),
            "val": len(split_ids["val"]),
            "test": len(split_ids["test"]),
            "total": len(split_ids["train"]) + len(split_ids["val"]) + len(split_ids["test"]),
        },
        "train": split_ids["train"],
        "val": split_ids["val"],
        "test": split_ids["test"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def make_split(
    dataset_root: str,
    output_path: Path,
    strategy: str,
    seed: int,
    ratios: Tuple[float, float, float],
) -> None:
    sample_ids = _collect_sample_ids(dataset_root)
    groups = _group_sample_ids(sample_ids, strategy)
    split_ids = _balanced_assign(groups, ratios, seed)

    name = output_path.stem
    _write_split_yaml(
        output_path=output_path,
        name=name,
        strategy=strategy,
        dataset_root=dataset_root,
        seed=seed,
        ratios=ratios,
        split_ids=split_ids,
    )

    print(f"[OK] {strategy}: {output_path}")
    print(
        f"      train={len(split_ids['train'])}, "
        f"val={len(split_ids['val'])}, test={len(split_ids['test'])}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MatPredict split YAMLs")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    out_dir = Path(args.output_dir)

    make_split(
        dataset_root=args.dataset_root,
        output_path=out_dir / "object_disjoint_v1.yaml",
        strategy="object_disjoint",
        seed=args.seed,
        ratios=ratios,
    )
    make_split(
        dataset_root=args.dataset_root,
        output_path=out_dir / "variance_disjoint_v1.yaml",
        strategy="variance_disjoint",
        seed=args.seed,
        ratios=ratios,
    )


if __name__ == "__main__":
    main()
