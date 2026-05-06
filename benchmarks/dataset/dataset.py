import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class Sample:
    image_path: Path
    albedo_path: Path
    orm_path: Path
    label_path: Path
    object_id: str
    var_id: str
    frame_id: str
    sample_id: str


def _is_valid_variant_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("_tmp"):
        return False
    return (path / "images").is_dir() and (path / "albedo").is_dir() and (path / "ORM").is_dir()


def _collect_variant_dirs(dataset_root: Path) -> List[Path]:
    variants: List[Path] = []
    for object_dir in sorted(dataset_root.iterdir()):
        if not object_dir.is_dir():
            continue
        for variant_dir in sorted(object_dir.iterdir()):
            if _is_valid_variant_dir(variant_dir):
                variants.append(variant_dir)
    return variants


def _collect_samples_from_variant(dataset_root: Path, variant_dir: Path) -> List[Sample]:
    image_dir = variant_dir / "images"
    albedo_dir = variant_dir / "albedo"
    orm_dir = variant_dir / "ORM"
    label_dir = variant_dir / "label"

    rel_variant = variant_dir.relative_to(dataset_root)
    object_id = rel_variant.parts[0]
    var_id = rel_variant.parts[1]

    samples: List[Sample] = []
    for image_path in sorted(image_dir.glob("*.png")):
        frame_id = image_path.stem
        albedo_path = albedo_dir / f"{frame_id}.png"
        orm_path = orm_dir / f"{frame_id}.png"
        label_path = label_dir / f"{frame_id}.png"

        # label may be missing in some cases; we will fallback to all-ones mask
        if not (albedo_path.exists() and orm_path.exists()):
            continue

        sample_id = f"{object_id}/{var_id}/{frame_id}"
        samples.append(
            Sample(
                image_path=image_path,
                albedo_path=albedo_path,
                orm_path=orm_path,
                label_path=label_path,
                object_id=object_id,
                var_id=var_id,
                frame_id=frame_id,
                sample_id=sample_id,
            )
        )
    return samples


def _collect_all_samples(dataset_root: Path) -> List[Sample]:
    all_samples: List[Sample] = []
    for variant_dir in _collect_variant_dirs(dataset_root):
        all_samples.extend(_collect_samples_from_variant(dataset_root, variant_dir))
    return all_samples


def _split_variant_dirs(
    variant_dirs: Sequence[Path],
    split: str,
    split_ratio: Tuple[float, float, float],
    seed: int,
) -> List[Path]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split}'. Use train/val/test.")

    if len(split_ratio) != 3:
        raise ValueError("split_ratio must have 3 values: (train, val, test).")

    ratio_sum = sum(split_ratio)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split_ratio should sum to 1.0, got {ratio_sum}.")

    dirs = list(variant_dirs)
    rng = random.Random(seed)
    rng.shuffle(dirs)

    n_total = len(dirs)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)
    n_test = n_total - n_train - n_val

    train_dirs = dirs[:n_train]
    val_dirs = dirs[n_train : n_train + n_val]
    test_dirs = dirs[n_train + n_val : n_train + n_val + n_test]

    table: Dict[str, List[Path]] = {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs,
    }
    return table[split]


def _load_split_ids(split_file: Path, split: str) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict) or split not in payload:
        raise ValueError(f"Split file {split_file} must contain key '{split}'.")

    ids_raw = payload.get(split)
    if not isinstance(ids_raw, list):
        raise ValueError(f"'{split}' in {split_file} must be a list of sample ids.")

    ids: List[str] = []
    for item in ids_raw:
        ids.append(str(item).strip().strip("/"))
    return ids


class MatPredictDataset(Dataset):
    """Loads object-shaped per-view PBR targets from multimat.

    Input:
      - images/{id}.png
    Targets:
      - albedo/{id}.png (3 channels)
      - ORM/{id}.png, using G as roughness and B as metallic (1+1 channels)
      - label/{id}.png -> material id map (1 channel, integer class id)
        and binary object mask (1 channel, 0/1)

    Supports two split modes:
      1) Official split file (recommended): split_file=...yaml
      2) Random variant split fallback: split_ratio + seed
    """

    def __init__(
        self,
        dataset_root: str,
        split: str,
        image_size: int = 224,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        use_imagenet_norm: bool = True,
        split_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Use train/val/test.")

        all_samples = _collect_all_samples(self.dataset_root)
        if not all_samples:
            raise RuntimeError(
                f"No valid samples found in {self.dataset_root}. "
                "Expected .../<object>/<varXXX>/{images,albedo,ORM}."
            )

        if split_file is not None:
            split_ids = _load_split_ids(Path(split_file), split)
            sample_index = {s.sample_id: s for s in all_samples}

            missing_ids = [sid for sid in split_ids if sid not in sample_index]
            if missing_ids:
                preview = ", ".join(missing_ids[:10])
                raise ValueError(
                    f"Found {len(missing_ids)} ids in split file but missing in dataset. "
                    f"First few: {preview}"
                )
            selected_samples = [sample_index[sid] for sid in split_ids]
        else:
            variant_dirs = _collect_variant_dirs(self.dataset_root)
            split_variant_dirs = _split_variant_dirs(variant_dirs, split, split_ratio, seed)
            if not split_variant_dirs:
                raise RuntimeError(f"Split '{split}' has no data. Please check split_ratio and dataset size.")

            selected_samples: List[Sample] = []
            for variant_dir in split_variant_dirs:
                selected_samples.extend(_collect_samples_from_variant(self.dataset_root, variant_dir))

        if not selected_samples:
            raise RuntimeError(f"Split '{split}' has no paired samples (images+albedo+ORM).")

        self.samples = selected_samples
        self.input_transform = self._build_input_transform(image_size, use_imagenet_norm)
        self.target_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self.mask_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )
        self.label_resize = T.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
        self.num_material_classes = self._infer_num_material_classes()

    @staticmethod
    def _build_input_transform(image_size: int, use_imagenet_norm: bool) -> T.Compose:
        transforms = [T.Resize((image_size, image_size)), T.ToTensor()]
        if use_imagenet_norm:
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        return T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def _infer_num_material_classes(self) -> int:
        max_label = 0
        for sample in self.samples:
            if not sample.label_path.exists():
                continue
            label_img = Image.open(sample.label_path).convert("L")
            label_resized = self.label_resize(label_img)
            label_tensor = torch.from_numpy(np.asarray(label_resized, dtype=np.int64))
            sample_max = int(label_tensor.max().item()) if label_tensor.numel() > 0 else 0
            if sample_max > max_label:
                max_label = sample_max
        return max_label + 1

    def __getitem__(self, index: int):
        sample = self.samples[index]

        input_img = Image.open(sample.image_path).convert("RGB")
        albedo_img = Image.open(sample.albedo_path).convert("RGB")
        orm_img = Image.open(sample.orm_path).convert("RGB")

        input_tensor = self.input_transform(input_img)
        albedo_tensor = self.target_transform(albedo_img)

        orm_tensor = self.target_transform(orm_img)
        roughness_tensor = orm_tensor[1:2, :, :]
        metallic_tensor = orm_tensor[2:3, :, :]

        if sample.label_path.exists():
            label_img = Image.open(sample.label_path).convert("L")
            label_img = self.label_resize(label_img)
            material_id = torch.from_numpy(np.asarray(label_img, dtype=np.int64))
            mask_tensor = (material_id.unsqueeze(0) > 0).float()
            has_material_label = torch.tensor(1.0, dtype=torch.float32)
        else:
            material_id = torch.zeros(
                (roughness_tensor.shape[-2], roughness_tensor.shape[-1]),
                dtype=torch.int64,
            )
            mask_tensor = torch.ones_like(roughness_tensor)
            has_material_label = torch.tensor(0.0, dtype=torch.float32)

        return {
            "input": input_tensor,
            "albedo": albedo_tensor,
            "roughness": roughness_tensor,
            "metallic": metallic_tensor,
            "mask": mask_tensor,
            "material_id": material_id,
            "has_material_label": has_material_label,
            "id": sample.sample_id,
        }
