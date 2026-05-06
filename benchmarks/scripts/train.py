import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import MatPredictDataset
from resnet50.model import ResNet50UNet
from swin_t.model import SwinTUNet
from utils import make_prediction_panel


DEFAULTS: Dict[str, Any] = {
    "dataset_root": None,
    "model_name": "resnet50_unet",
    "output_dir": "./outputs",
    "split_file": None,
    "pretrained_model": None,
    "image_size": 224,
    "batch_size": 8,
    "epochs": 50,
    "num_workers": 4,
    "validate_every": 1,
    "num_vis_samples": 4,
    "val_vis_ids_file": None,
    "log_every_steps": 50,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "optimizer_name": "adamw",
    "optimizer_momentum": 0.9,
    "albedo_weight": 1.0,
    "roughness_weight": 1.0,
    "metallic_weight": 1.0,
    "material_weight": 1.0,
    "material_ce_weight": 0.0,
    "task_mode": "pbr",
    "material_num_classes": None,
    "material_ignore_background": True,
    "material_map_file": None,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
    "no_pretrained": False,
    "use_wandb": False,
    "wandb_project": "matpredict_segmentation",
    "wandb_run_name": None,
    "append_timestamp": True,
    "timestamp_format": "%Y%m%d_%H%M%S",
    "log_path": None,
}


MATERIAL_SEGMENTATION_MAP: Dict[str, int] = {
    "background": 0,
    "concrete": 1,
    "fabric": 2,
    "leather": 3,
    "metal": 4,
    "plastic": 5,
    "stone": 6,
    "wood": 7,
}

_MATERIAL_COLORS: List[Tuple[int, int, int]] = [
    (30, 30, 30),      # background
    (230, 85, 13),     # concrete
    (49, 130, 189),    # fabric
    (227, 26, 28),     # leather
    (106, 61, 154),    # metal
    (255, 127, 0),     # plastic
    (177, 89, 40),     # stone
    (51, 160, 44),     # wood
]


def _load_config(config_path: str) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a dict, got: {type(payload)}")
    return payload


def _build_parser(config_defaults: Dict[str, Any], required_dataset_root: bool) -> argparse.ArgumentParser:
    merged = dict(DEFAULTS)
    merged.update(config_defaults)

    parser = argparse.ArgumentParser(description="Train segmentation benchmarks (ResNet50 U-Net / Swin-T U-Net) for object PBR maps")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config. CLI args override config values.")

    parser.add_argument("--dataset_root", type=str, required=required_dataset_root, default=merged["dataset_root"], help="Path to MatPredictDataset root")
    parser.add_argument("--model_name", type=str, default=merged["model_name"], choices=["resnet50_unet", "swin_t_unet"])
    parser.add_argument("--output_dir", type=str, default=merged["output_dir"])

    parser.add_argument("--split_file", type=str, default=merged["split_file"], help="Path to official split YAML")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=merged["pretrained_model"],
        help="Checkpoint path for resuming training. Use null/'' to disable.",
    )

    parser.add_argument("--image_size", type=int, default=merged["image_size"])
    parser.add_argument("--batch_size", type=int, default=merged["batch_size"])
    parser.add_argument("--epochs", type=int, default=merged["epochs"])
    parser.add_argument("--num_workers", type=int, default=merged["num_workers"])

    parser.add_argument("--validate_every", type=int, default=merged["validate_every"])
    parser.add_argument("--num_vis_samples", type=int, default=merged["num_vis_samples"])
    parser.add_argument(
        "--val_vis_ids_file",
        type=str,
        default=merged["val_vis_ids_file"],
        help="Optional YAML file with fixed val visualization sample ids.",
    )
    parser.add_argument("--log_every_steps", type=int, default=merged["log_every_steps"], help="Log train metrics to W&B every N optimizer steps.")

    parser.add_argument("--lr", type=float, default=merged["lr"])
    parser.add_argument("--weight_decay", type=float, default=merged["weight_decay"])
    parser.add_argument("--optimizer_name", type=str, default=merged["optimizer_name"], choices=["adamw", "adam", "sgd"])
    parser.add_argument("--optimizer_momentum", type=float, default=merged["optimizer_momentum"], help="Used only when optimizer_name=sgd")
    parser.add_argument("--albedo_weight", type=float, default=merged["albedo_weight"])
    parser.add_argument("--roughness_weight", type=float, default=merged["roughness_weight"])
    parser.add_argument("--metallic_weight", type=float, default=merged["metallic_weight"])
    parser.add_argument("--material_weight", type=float, default=merged["material_weight"])
    parser.add_argument(
        "--material_ce_weight",
        type=float,
        default=merged["material_ce_weight"],
        help="Cross-entropy weight for task_mode=material. Use with material_weight (IoU loss).",
    )
    parser.add_argument("--task_mode", type=str, default=merged["task_mode"], choices=["pbr", "material"])
    parser.add_argument(
        "--material_num_classes",
        type=int,
        default=merged["material_num_classes"],
        help="Material class count for task_mode=material. If unset, infer from train split labels.",
    )
    parser.add_argument(
        "--material_ignore_background",
        action=argparse.BooleanOptionalAction,
        default=bool(merged["material_ignore_background"]),
        help="Exclude class 0 from IoU loss/metrics in task_mode=material.",
    )
    parser.add_argument(
        "--material_map_file",
        type=str,
        default=merged["material_map_file"],
        help="Optional YAML material id map {class_name: id}. Used for color legend and reporting.",
    )

    parser.add_argument("--train_ratio", type=float, default=merged["train_ratio"])
    parser.add_argument("--val_ratio", type=float, default=merged["val_ratio"])
    parser.add_argument("--test_ratio", type=float, default=merged["test_ratio"])
    parser.add_argument("--seed", type=int, default=merged["seed"])

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        default=bool(merged["no_pretrained"]),
        help="Disable ImageNet backbone initialization.",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=bool(merged["use_wandb"]),
    )
    parser.add_argument("--wandb_project", type=str, default=merged["wandb_project"])
    parser.add_argument("--wandb_run_name", type=str, default=merged["wandb_run_name"])
    parser.add_argument(
        "--append_timestamp",
        action=argparse.BooleanOptionalAction,
        default=bool(merged["append_timestamp"]),
        help="Append timestamp to output_dir and wandb_run_name.",
    )
    parser.add_argument("--timestamp_format", type=str, default=merged["timestamp_format"])
    parser.add_argument("--log_path", type=str, default=merged["log_path"], help="Optional training log file path. Relative paths are under output_dir.")

    return parser


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args()

    config_defaults: Dict[str, Any] = {}
    if pre_args.config:
        config_defaults = _load_config(pre_args.config)

    required_dataset_root = "dataset_root" not in config_defaults or config_defaults.get("dataset_root") in (None, "")
    parser = _build_parser(config_defaults, required_dataset_root=required_dataset_root)
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    return args


def _timestamp_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def _append_timestamp_name(base: str, ts: str) -> str:
    if "{ts}" in base:
        return base.replace("{ts}", ts)
    return f"{base}_{ts}"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
        return len(data)

    def flush(self):
        for st in self.streams:
            st.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    task_mode: str,
    pretrained: bool = True,
    image_size: int = 224,
    material_num_classes: int = 0,
) -> nn.Module:
    enable_pbr_head = task_mode == "pbr"
    num_material_classes = material_num_classes if task_mode == "material" else 0

    if model_name == "resnet50_unet":
        return ResNet50UNet(
            out_channels=5,
            pretrained=pretrained,
            num_material_classes=num_material_classes,
            enable_pbr_head=enable_pbr_head,
        )
    if model_name == "swin_t_unet":
        return SwinTUNet(
            out_channels=5,
            pretrained=pretrained,
            image_size=image_size,
            num_material_classes=num_material_classes,
            enable_pbr_head=enable_pbr_head,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _format_flops(flops: Optional[float]) -> str:
    if flops is None:
        return "unavailable"
    return f"{flops / 1e9:.3f} GFLOPs"


def _build_optimizer(args, model: nn.Module) -> optim.Optimizer:
    name = str(args.optimizer_name).lower()
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.optimizer_momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
    raise ValueError(f"Unsupported optimizer_name: {args.optimizer_name}")


def _build_lpips_model(device: torch.device):
    try:
        import lpips
    except Exception:
        return None

    m = lpips.LPIPS(net="alex")
    m = m.to(device)
    m.eval()
    return m


def _masked_lpips_albedo(
    pred_albedo: torch.Tensor,
    gt_albedo: torch.Tensor,
    mask3: torch.Tensor,
    lpips_model,
) -> Optional[torch.Tensor]:
    if lpips_model is None:
        return None

    pred_vis = pred_albedo * mask3 + (1.0 - mask3)
    gt_vis = gt_albedo * mask3 + (1.0 - mask3)

    pred_lp = pred_vis * 2.0 - 1.0
    gt_lp = gt_vis * 2.0 - 1.0
    with torch.no_grad():
        vals = lpips_model(pred_lp, gt_lp).view(-1)
    return vals.mean()


def _estimate_model_flops(model: nn.Module, image_size: int, device: torch.device) -> Optional[float]:
    x = torch.randn(1, 3, image_size, image_size, device=device)
    training = model.training
    model.eval()
    try:
        try:
            from fvcore.nn import FlopCountAnalysis
            with torch.no_grad():
                flops = float(FlopCountAnalysis(model, x).total())
            return flops
        except Exception:
            try:
                from thop import profile
                with torch.no_grad():
                    flops, _ = profile(model, inputs=(x,), verbose=False)
                return float(flops)
            except Exception:
                return None
    finally:
        model.train(training)


def _normalize_pretrained_model_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none" or s.lower() == "null":
        return None
    return s


def _load_material_segmentation_map(path: Optional[str]) -> Dict[str, int]:
    if path is None or str(path).strip() == "":
        return dict(MATERIAL_SEGMENTATION_MAP)

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"material_map_file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if payload is None:
        return dict(MATERIAL_SEGMENTATION_MAP)
    if not isinstance(payload, dict):
        raise ValueError(f"material_map_file must be a dict {{name: id}}, got: {type(payload)}")
    if "material_segmentation_map" in payload:
        payload = payload["material_segmentation_map"]
        if not isinstance(payload, dict):
            raise ValueError("material_segmentation_map must be a dict {name: id}.")

    name_to_id: Dict[str, int] = {}
    for k, v in payload.items():
        name_to_id[str(k)] = int(v)
    return name_to_id


def _build_material_id_to_name(name_to_id: Dict[str, int], num_classes: int) -> Dict[int, str]:
    id_to_name = {int(v): str(k) for k, v in name_to_id.items()}
    for cid in range(num_classes):
        if cid not in id_to_name:
            id_to_name[cid] = f"class_{cid}"
    return id_to_name


def _build_material_palette(num_classes: int) -> np.ndarray:
    colors: List[Tuple[int, int, int]] = []
    for cid in range(num_classes):
        if cid < len(_MATERIAL_COLORS):
            colors.append(_MATERIAL_COLORS[cid])
        else:
            colors.append(
                (
                    (37 * cid + 79) % 256,
                    (97 * cid + 31) % 256,
                    (17 * cid + 191) % 256,
                )
            )
    return np.asarray(colors, dtype=np.uint8)


def _input_to_uint8(input_tensor: torch.Tensor) -> np.ndarray:
    x = input_tensor.detach().cpu().float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x * std + mean).clamp(0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _colorize_material_ids(material_ids: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = material_ids.shape
    clipped = np.clip(material_ids, 0, len(palette) - 1)
    colored = palette[clipped.reshape(-1)].reshape(h, w, 3)
    return colored


def _confusion_matrix(
    pred_ids: torch.Tensor,
    gt_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=pred_ids.device)

    g = gt_ids[valid_mask].long()
    p = pred_ids[valid_mask].long()
    idx = g * num_classes + p
    hist = torch.bincount(idx, minlength=num_classes * num_classes).double()
    return hist.view(num_classes, num_classes)


def _material_metrics_from_confusion(
    conf: torch.Tensor,
    ignore_background: bool,
    eps: float = 1e-6,
) -> Dict[str, float]:
    total = conf.sum()
    diag = torch.diag(conf)
    gt_count = conf.sum(dim=1)
    pred_count = conf.sum(dim=0)
    union = gt_count + pred_count - diag
    dice_den = gt_count + pred_count

    class_start = 1 if ignore_background else 0
    valid_iou = union[class_start:] > eps
    valid_acc = gt_count[class_start:] > eps
    valid_dice = dice_den[class_start:] > eps

    iou = torch.zeros_like(union)
    iou[union > eps] = diag[union > eps] / union[union > eps]
    class_acc = torch.zeros_like(gt_count)
    class_acc[gt_count > eps] = diag[gt_count > eps] / gt_count[gt_count > eps]
    dice = torch.zeros_like(dice_den)
    dice[dice_den > eps] = (2.0 * diag[dice_den > eps]) / dice_den[dice_den > eps]

    if valid_iou.any():
        miou = float(iou[class_start:][valid_iou].mean().item())
        freq = gt_count[class_start:][valid_iou]
        fwiou = float(((freq / freq.sum().clamp_min(eps)) * iou[class_start:][valid_iou]).sum().item())
    else:
        miou = 1.0
        fwiou = 1.0

    if valid_acc.any():
        macc = float(class_acc[class_start:][valid_acc].mean().item())
    else:
        macc = 1.0

    if valid_dice.any():
        mdice = float(dice[class_start:][valid_dice].mean().item())
    else:
        mdice = 1.0

    pixel_acc = float((diag.sum() / total.clamp_min(eps)).item()) if total > 0 else 1.0
    return {
        "miou_material": miou,
        "fwiou_material": fwiou,
        "pixel_acc_material": pixel_acc,
        "macc_material": macc,
        "mdice_material": mdice,
    }


def _make_material_panel(
    input_tensor: torch.Tensor,
    pred_ids: torch.Tensor,
    gt_ids: torch.Tensor,
    sample_miou: float,
    palette: np.ndarray,
) -> Image.Image:
    inp = _input_to_uint8(input_tensor)
    pred_np = pred_ids.detach().cpu().numpy().astype(np.int64)
    gt_np = gt_ids.detach().cpu().numpy().astype(np.int64)
    pred_col = _colorize_material_ids(pred_np, palette)
    gt_col = _colorize_material_ids(gt_np, palette)

    panel_np = np.concatenate([inp, pred_col, gt_col], axis=1)
    panel = Image.fromarray(panel_np, mode="RGB")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"IoU={sample_miou:.4f} | input | pred | gt", fill=(255, 255, 255))
    return panel


def _masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    abs_err = (pred - gt).abs() * mask
    denom = mask.sum().clamp_min(eps)
    return abs_err.sum() / denom


def _masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sq_err = (pred - gt).pow(2) * mask
    denom = mask.sum().clamp_min(eps)
    return sq_err.sum() / denom


def _masked_global_ssim(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2

    b, c, _, _ = pred.shape
    scores = []
    for bi in range(b):
        for ci in range(c):
            x = pred[bi, ci]
            y = gt[bi, ci]
            m = mask[bi, ci]
            wsum = m.sum()
            if wsum <= eps:
                continue

            mu_x = (x * m).sum() / wsum
            mu_y = (y * m).sum() / wsum
            var_x = ((x - mu_x).pow(2) * m).sum() / wsum
            var_y = ((y - mu_y).pow(2) * m).sum() / wsum
            cov_xy = (((x - mu_x) * (y - mu_y)) * m).sum() / wsum

            num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
            den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
            scores.append(num / (den + eps))

    if not scores:
        return torch.tensor(1.0, device=pred.device)
    return torch.stack(scores).mean()


def _metrics_from_preds(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    mse = _masked_mse(pred, gt, mask)
    ssim = _masked_global_ssim(pred, gt, mask)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
    return {
        "mse": float(mse.item()),
        "ssim": float(ssim.item()),
        "psnr": float(psnr.item()),
    }


def _extract_model_outputs(model_output) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if isinstance(model_output, dict):
        return model_output.get("pbr"), model_output.get("material_logits")
    if torch.is_tensor(model_output):
        return model_output, None
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def compute_pbr_loss(
    pred_pbr: torch.Tensor,
    albedo_gt: torch.Tensor,
    roughness_gt: torch.Tensor,
    metallic_gt: torch.Tensor,
    object_mask: torch.Tensor,
    albedo_weight: float,
    roughness_weight: float,
    metallic_weight: float,
):
    # Keep model head unconstrained and map logits to physical [0, 1] range
    # only when computing PBR supervision.
    pred_pbr = torch.sigmoid(pred_pbr)
    pred_albedo = pred_pbr[:, :3, :, :]
    pred_roughness = pred_pbr[:, 3:4, :, :]
    pred_metallic = pred_pbr[:, 4:5, :, :]

    albedo_mask = object_mask.repeat(1, 3, 1, 1)
    rough_mask = object_mask

    loss_albedo = _masked_l1(pred_albedo, albedo_gt, albedo_mask)
    loss_roughness = _masked_l1(pred_roughness, roughness_gt, rough_mask)
    loss_metallic = _masked_l1(pred_metallic, metallic_gt, rough_mask)
    loss = (
        albedo_weight * loss_albedo
        + roughness_weight * loss_roughness
        + metallic_weight * loss_metallic
    )

    return loss, loss_albedo.detach(), loss_roughness.detach(), loss_metallic.detach()


@torch.no_grad()
def _hard_miou(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    ignore_background: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    ious = []
    class_start = 1 if ignore_background else 0
    for c in range(class_start, num_classes):
        pred_c = (pred == c) & valid_mask
        tgt_c = (target == c) & valid_mask
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union > 0:
            ious.append((inter + eps) / (union + eps))
    if not ious:
        return torch.tensor(1.0, device=logits.device)
    return torch.stack(ious).mean()


def _material_iou_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    ignore_background: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)

    target_oh = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid_mask.unsqueeze(1).float()

    inter = (probs * target_oh * valid).sum(dim=(0, 2, 3))
    union = ((probs + target_oh - probs * target_oh) * valid).sum(dim=(0, 2, 3))

    class_start = 1 if ignore_background else 0
    inter = inter[class_start:]
    union = union[class_start:]
    valid_classes = union > eps
    if valid_classes.any():
        iou = (inter[valid_classes] + eps) / (union[valid_classes] + eps)
        return 1.0 - iou.mean()
    return torch.tensor(0.0, device=logits.device)


def _material_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    ignore_background: bool,
    ignore_index: int = -100,
) -> torch.Tensor:
    train_target = target.clone()
    ignore_mask = ~valid_mask
    if ignore_background:
        ignore_mask = ignore_mask | (target == 0)
    train_target = train_target.masked_fill(ignore_mask, ignore_index)

    if (train_target != ignore_index).sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, train_target, ignore_index=ignore_index)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    task_mode: str,
    albedo_weight: float,
    roughness_weight: float,
    metallic_weight: float,
    material_weight: float,
    material_ce_weight: float,
    material_ignore_background: bool,
    lpips_model=None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0

    total_albedo = 0.0
    total_roughness = 0.0
    total_metallic = 0.0
    total_mse_albedo = 0.0
    total_ssim_albedo = 0.0
    total_psnr_albedo = 0.0
    total_lpips_albedo = 0.0
    total_mse_rough = 0.0
    total_ssim_rough = 0.0
    total_psnr_rough = 0.0
    total_mse_metal = 0.0
    total_ssim_metal = 0.0
    total_psnr_metal = 0.0

    total_material_iou_loss = 0.0
    total_material_ce_loss = 0.0
    total_material_count = 0
    total_material_conf = None
    material_num_classes_eval: Optional[int] = None
    n = 0

    for batch in loader:
        x = batch["input"].to(device)
        out = model(x)
        pred_pbr, pred_material = _extract_model_outputs(out)
        bs = x.size(0)

        if task_mode == "pbr":
            y_albedo = batch["albedo"].to(device)
            y_roughness = batch["roughness"].to(device)
            y_metallic = batch["metallic"].to(device)
            y_mask = batch["mask"].to(device)

            if pred_pbr is None:
                raise RuntimeError("Model did not return PBR prediction in task_mode='pbr'.")

            loss, loss_albedo, loss_roughness, loss_metallic = compute_pbr_loss(
                pred_pbr,
                y_albedo,
                y_roughness,
                y_metallic,
                y_mask,
                albedo_weight,
                roughness_weight,
                metallic_weight,
            )

            pred_albedo = pred_pbr[:, :3, :, :].clamp(0.0, 1.0)
            pred_rough = pred_pbr[:, 3:4, :, :].clamp(0.0, 1.0)
            pred_metal = pred_pbr[:, 4:5, :, :].clamp(0.0, 1.0)
            mask3 = y_mask.repeat(1, 3, 1, 1)
            m_albedo = _metrics_from_preds(pred_albedo, y_albedo, mask3)
            m_rough = _metrics_from_preds(pred_rough, y_roughness, y_mask)
            m_metal = _metrics_from_preds(pred_metal, y_metallic, y_mask)
            m_lpips = _masked_lpips_albedo(pred_albedo, y_albedo, mask3, lpips_model)

            total_loss += float(loss.item()) * bs
            total_albedo += float(loss_albedo.item()) * bs
            total_roughness += float(loss_roughness.item()) * bs
            total_metallic += float(loss_metallic.item()) * bs
            total_mse_albedo += m_albedo["mse"] * bs
            total_ssim_albedo += m_albedo["ssim"] * bs
            total_psnr_albedo += m_albedo["psnr"] * bs
            if m_lpips is not None:
                total_lpips_albedo += float(m_lpips.item()) * bs
            total_mse_rough += m_rough["mse"] * bs
            total_ssim_rough += m_rough["ssim"] * bs
            total_psnr_rough += m_rough["psnr"] * bs
            total_mse_metal += m_metal["mse"] * bs
            total_ssim_metal += m_metal["ssim"] * bs
            total_psnr_metal += m_metal["psnr"] * bs
        else:
            if pred_material is None:
                if pred_pbr is not None and pred_pbr.shape[1] > 1:
                    pred_material = pred_pbr
                else:
                    raise RuntimeError("Model did not return material logits in task_mode='material'.")

            y_mat = batch["material_id"].to(device).long()
            material_num_classes_eval = int(pred_material.shape[1])
            has_mat = batch["has_material_label"].to(device).view(-1, 1, 1)
            valid_mask = (has_mat > 0.5).expand_as(y_mat)

            iou_loss = _material_iou_loss(
                pred_material,
                y_mat,
                valid_mask=valid_mask,
                ignore_background=material_ignore_background,
            )
            ce_loss = _material_cross_entropy_loss(
                pred_material,
                y_mat,
                valid_mask=valid_mask,
                ignore_background=material_ignore_background,
            )
            loss = material_weight * iou_loss + material_ce_weight * ce_loss

            total_loss += float(loss.item()) * bs
            total_material_iou_loss += float(iou_loss.item()) * bs
            total_material_ce_loss += float(ce_loss.item()) * bs
            total_material_count += bs
            pred_mat = pred_material.argmax(dim=1)
            conf_batch = _confusion_matrix(
                pred_ids=pred_mat,
                gt_ids=y_mat,
                valid_mask=valid_mask,
                num_classes=pred_material.shape[1],
            )
            if total_material_conf is None:
                total_material_conf = conf_batch
            else:
                total_material_conf = total_material_conf + conf_batch

        n += bs

    if task_mode == "pbr":
        return {
            "loss": total_loss / max(n, 1),
            "loss_albedo": total_albedo / max(n, 1),
            "loss_roughness": total_roughness / max(n, 1),
            "loss_metallic": total_metallic / max(n, 1),
            "mse_albedo": total_mse_albedo / max(n, 1),
            "ssim_albedo": total_ssim_albedo / max(n, 1),
            "psnr_albedo": total_psnr_albedo / max(n, 1),
            "lpips_albedo": (total_lpips_albedo / max(n, 1)) if lpips_model is not None else float("nan"),
            "mse_roughness": total_mse_rough / max(n, 1),
            "ssim_roughness": total_ssim_rough / max(n, 1),
            "psnr_roughness": total_psnr_rough / max(n, 1),
            "mse_metallic": total_mse_metal / max(n, 1),
            "ssim_metallic": total_ssim_metal / max(n, 1),
            "psnr_metallic": total_psnr_metal / max(n, 1),
        }

    mat_metrics = _material_metrics_from_confusion(
        total_material_conf
        if total_material_conf is not None
        else torch.zeros(
            (material_num_classes_eval or 1, material_num_classes_eval or 1),
            device=device,
            dtype=torch.float64,
        ),
        ignore_background=material_ignore_background,
    )
    return {
        "loss": total_loss / max(n, 1),
        "loss_material_iou": total_material_iou_loss / max(total_material_count, 1),
        "loss_material_ce": total_material_ce_loss / max(total_material_count, 1),
        **mat_metrics,
    }


@torch.no_grad()
def collect_visual_panels(
    model,
    loader,
    device,
    max_samples: int,
    lpips_model=None,
    selected_ids: Optional[List[str]] = None,
) -> List[Tuple[str, object]]:
    model.eval()
    panels: List[Tuple[str, object]] = []
    selected_set = set(selected_ids) if selected_ids else None
    selected_index = {sid: i for i, sid in enumerate(selected_ids)} if selected_ids else None
    if max_samples <= 0:
        return panels

    for batch in loader:
        x = batch["input"].to(device)
        y_albedo = batch["albedo"].to(device)
        y_roughness = batch["roughness"].to(device)
        y_metallic = batch["metallic"].to(device)
        y_mask = batch["mask"].to(device)
        ids = batch["id"]

        out = model(x)
        pred_pbr, _ = _extract_model_outputs(out)
        if pred_pbr is None:
            raise RuntimeError("collect_visual_panels requires a PBR head output.")
        pred_albedo = pred_pbr[:, :3, :, :].clamp(0.0, 1.0)
        pred_rough = pred_pbr[:, 3:4, :, :].clamp(0.0, 1.0)
        pred_metal = pred_pbr[:, 4:5, :, :].clamp(0.0, 1.0)

        # For visualization, enforce object-only prediction display.
        inv_mask = 1.0 - y_mask
        pred_albedo_vis = pred_albedo * y_mask.repeat(1, 3, 1, 1) + inv_mask.repeat(1, 3, 1, 1)
        pred_rough_vis = pred_rough * y_mask + inv_mask
        pred_metal_vis = pred_metal * y_mask + inv_mask

        bs = x.size(0)
        for i in range(bs):
            if selected_set is not None and ids[i] not in selected_set:
                continue
            m_albedo = _metrics_from_preds(
                pred_albedo[i:i+1],
                y_albedo[i:i+1],
                y_mask[i:i+1].repeat(1, 3, 1, 1),
            )
            m_rough = _metrics_from_preds(
                pred_rough[i:i+1],
                y_roughness[i:i+1],
                y_mask[i:i+1],
            )
            m_metal = _metrics_from_preds(
                pred_metal[i:i+1],
                y_metallic[i:i+1],
                y_mask[i:i+1],
            )
            m_lpips = _masked_lpips_albedo(
                pred_albedo[i:i+1],
                y_albedo[i:i+1],
                y_mask[i:i+1].repeat(1, 3, 1, 1),
                lpips_model,
            )
            if m_lpips is not None:
                m_albedo["lpips"] = float(m_lpips.item())

            panel = make_prediction_panel(
                input_img=x[i],
                pred_albedo=pred_albedo_vis[i],
                gt_albedo=y_albedo[i],
                pred_roughness=pred_rough_vis[i],
                gt_roughness=y_roughness[i],
                pred_metallic=pred_metal_vis[i],
                gt_metallic=y_metallic[i],
                input_is_normalized=True,
                albedo_metrics=m_albedo,
                roughness_metrics=m_rough,
                metallic_metrics=m_metal,
                object_mask=y_mask[i],
            )
            panels.append((ids[i], panel))
            if len(panels) >= max_samples:
                break
        if len(panels) >= max_samples:
            break

    if selected_index is not None:
        panels.sort(key=lambda x: selected_index.get(x[0], 10**9))
    return panels[:max_samples]


@torch.no_grad()
def collect_material_visual_panels(
    model,
    loader,
    device,
    max_samples: int,
    material_palette: np.ndarray,
    ignore_background: bool,
    selected_ids: Optional[List[str]] = None,
) -> List[Tuple[str, object]]:
    model.eval()
    panels: List[Tuple[str, object]] = []
    selected_set = set(selected_ids) if selected_ids else None
    selected_index = {sid: i for i, sid in enumerate(selected_ids)} if selected_ids else None
    if max_samples <= 0:
        return panels

    for batch in loader:
        x = batch["input"].to(device)
        y_mat = batch["material_id"].to(device).long()
        has_mat = batch["has_material_label"].to(device).view(-1, 1, 1)
        ids = batch["id"]

        out = model(x)
        pred_pbr, pred_material = _extract_model_outputs(out)
        if pred_material is None:
            if pred_pbr is not None and pred_pbr.shape[1] > 1:
                pred_material = pred_pbr
            else:
                raise RuntimeError("collect_material_visual_panels requires material logits.")

        pred_ids = pred_material.argmax(dim=1)
        bs = x.size(0)
        for i in range(bs):
            if selected_set is not None and ids[i] not in selected_set:
                continue
            if float(has_mat[i].item()) <= 0.0:
                continue
            valid_i = torch.ones_like(y_mat[i], dtype=torch.bool, device=y_mat.device)
            conf_i = _confusion_matrix(
                pred_ids=pred_ids[i],
                gt_ids=y_mat[i],
                valid_mask=valid_i,
                num_classes=pred_material.shape[1],
            )
            miou_i = _material_metrics_from_confusion(conf_i, ignore_background=ignore_background)["miou_material"]
            panel = _make_material_panel(
                input_tensor=x[i],
                pred_ids=pred_ids[i],
                gt_ids=y_mat[i],
                sample_miou=miou_i,
                palette=material_palette,
            )
            panels.append((ids[i], panel))
            if len(panels) >= max_samples:
                break
        if len(panels) >= max_samples:
            break
    if selected_index is not None:
        panels.sort(key=lambda x: selected_index.get(x[0], 10**9))
    return panels[:max_samples]


def _save_visual_panels(vis_dir: Path, split_name: str, epoch: int, panels: List[Tuple[str, object]]) -> None:
    split_dir = vis_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx, (sample_id, panel) in enumerate(panels):
        safe_id = sample_id.replace("/", "__")
        panel.save(split_dir / f"epoch_{epoch:03d}_{idx:02d}_{safe_id}.png")


def _load_vis_ids(vis_ids_file: Path) -> List[str]:
    if not vis_ids_file.exists():
        raise FileNotFoundError(f"val_vis_ids_file not found: {vis_ids_file}")
    with vis_ids_file.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if isinstance(payload, list):
        return [str(x).strip().strip("/") for x in payload]
    if isinstance(payload, dict) and "sample_ids" in payload and isinstance(payload["sample_ids"], list):
        return [str(x).strip().strip("/") for x in payload["sample_ids"]]
    raise ValueError(f"Unsupported val_vis_ids_file format: {vis_ids_file}")


def _save_used_config(args, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "used_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)


def _build_checkpoint_payload(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    args,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Optional[Dict[str, float]],
    best_val: float,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def _resume_if_requested(model, optimizer, scheduler, args, device) -> Tuple[int, float]:
    ckpt_path_raw = _normalize_pretrained_model_arg(args.pretrained_model)
    if ckpt_path_raw is None:
        return 1, float("inf")

    ckpt_path = Path(ckpt_path_raw)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"pretrained_model checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "best_val" in ckpt:
            best_val = float(ckpt["best_val"])
        elif "val_metrics" in ckpt and isinstance(ckpt["val_metrics"], dict) and "loss" in ckpt["val_metrics"]:
            best_val = float(ckpt["val_metrics"]["loss"])
        else:
            best_val = float("inf")
    else:
        model.load_state_dict(ckpt, strict=True)
        start_epoch = 1
        best_val = float("inf")

    print(f"Resumed from checkpoint: {ckpt_path}")
    print(f"Resume start epoch: {start_epoch}")
    return start_epoch, best_val


def train(args):
    ts = _timestamp_now(args.timestamp_format)
    setattr(args, "run_timestamp", ts)

    if args.append_timestamp:
        args.output_dir = _append_timestamp_name(str(args.output_dir), ts)
        if args.wandb_run_name is None or str(args.wandb_run_name).strip() == "":
            args.wandb_run_name = _append_timestamp_name(args.model_name, ts)
        else:
            args.wandb_run_name = _append_timestamp_name(str(args.wandb_run_name), ts)
    else:
        if args.use_wandb and (args.wandb_run_name is None or str(args.wandb_run_name).strip() == ""):
            args.wandb_run_name = args.model_name

    set_seed(args.seed)

    train_ds = MatPredictDataset(
        dataset_root=args.dataset_root,
        split="train",
        image_size=args.image_size,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        use_imagenet_norm=True,
        split_file=args.split_file,
    )
    val_ds = MatPredictDataset(
        dataset_root=args.dataset_root,
        split="val",
        image_size=args.image_size,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        use_imagenet_norm=True,
        split_file=args.split_file,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    val_vis_ids: Optional[List[str]] = None
    if args.val_vis_ids_file is not None and str(args.val_vis_ids_file).strip() != "":
        val_vis_ids = _load_vis_ids(Path(args.val_vis_ids_file))
        val_id_set = {s.sample_id for s in val_ds.samples}
        missing = [sid for sid in val_vis_ids if sid not in val_id_set]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"{len(missing)} val_vis_ids not found in val split. First few: {preview}")

    inferred_material_num_classes = max(int(train_ds.num_material_classes), int(val_ds.num_material_classes))
    material_num_classes = int(args.material_num_classes) if args.material_num_classes is not None else inferred_material_num_classes
    if args.task_mode == "material" and material_num_classes < 2:
        raise ValueError(
            "task_mode=material requires at least 2 classes (background + 1 material). "
            f"Got material_num_classes={material_num_classes}"
        )
    setattr(args, "resolved_material_num_classes", material_num_classes)
    material_name_to_id = _load_material_segmentation_map(args.material_map_file)
    material_id_to_name = _build_material_id_to_name(material_name_to_id, material_num_classes)
    material_palette = _build_material_palette(material_num_classes)
    setattr(args, "resolved_material_name_to_id", material_name_to_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model_name,
        task_mode=args.task_mode,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        material_num_classes=material_num_classes,
    ).to(device)
    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_flops = _estimate_model_flops(model, args.image_size, device)
    lpips_model = _build_lpips_model(device) if args.task_mode == "pbr" else None

    optimizer = _build_optimizer(args, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val = _resume_if_requested(model, optimizer, scheduler, args, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_used_config(args, out_dir)

    resolved_log_path = None
    if args.log_path is not None and str(args.log_path).strip() != "":
        raw_log_path = str(args.log_path)
        if "{ts}" in raw_log_path:
            raw_log_path = raw_log_path.replace("{ts}", ts)
        lp = Path(raw_log_path)
        if not lp.is_absolute():
            lp = out_dir / lp
        lp.parent.mkdir(parents=True, exist_ok=True)
        log_f = lp.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.stdout, log_f)
        sys.stderr = _Tee(sys.stderr, log_f)
        resolved_log_path = str(lp)
        setattr(args, "resolved_log_path", resolved_log_path)
        print(f"Logging to file: {resolved_log_path}")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = args.use_wandb
    wb = None
    if use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("--use_wandb is set but wandb is not installed. Please install wandb first.") from exc

        wb = wandb
        wb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        wb.define_metric("epoch")
        wb.define_metric("step")
        wb.define_metric("train/*", step_metric="epoch")

    print(f"Device: {device}")
    if args.config:
        print(f"Loaded config: {args.config}")
    if args.split_file:
        print(f"Using split file: {args.split_file}")
    else:
        print(
            "Using random fallback split ratio "
            f"({args.train_ratio}, {args.val_ratio}, {args.test_ratio}) with seed={args.seed}"
        )

    ckpt_mode = _normalize_pretrained_model_arg(args.pretrained_model)
    if ckpt_mode is None:
        print("pretrained_model: None (no checkpoint resume)")
    else:
        print(f"pretrained_model: {ckpt_mode}")

    print(f"Output dir: {args.output_dir}")
    if resolved_log_path is not None:
        print(f"Log path: {resolved_log_path}")
    if args.use_wandb:
        print(f"W&B run name: {args.wandb_run_name}")
        print(f"W&B log_every_steps: {args.log_every_steps}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    if val_vis_ids is not None:
        print(f"Using fixed val visualization ids: {len(val_vis_ids)} from {args.val_vis_ids_file}")
    print(f"Task mode: {args.task_mode}")
    if args.task_mode == "material":
        print(f"Material classes: {material_num_classes} (ignore_background={args.material_ignore_background})")
        print(f"Material loss weights: iou={args.material_weight}, ce={args.material_ce_weight}")
        print(f"Material id map: {material_id_to_name}")
    print(f"Model params (total): {num_params_total:,}")
    print(f"Model params (trainable): {num_params_trainable:,}")
    print(f"Model FLOPs (1x3x{args.image_size}x{args.image_size}): {_format_flops(est_flops)}")
    print(f"Optimizer: {args.optimizer_name} (lr={args.lr}, weight_decay={args.weight_decay}, momentum={args.optimizer_momentum})")
    if args.task_mode == "pbr":
        print(f"LPIPS(albedo): {'enabled' if lpips_model is not None else 'unavailable (install lpips)'}")

    if start_epoch > args.epochs:
        print(f"Checkpoint epoch already exceeds target epochs ({start_epoch-1} >= {args.epochs}), nothing to train.")
        if wb is not None:
            wb.finish()
        return

    global_step = (start_epoch - 1) * len(train_loader)
    prev_logged_train_loss: Optional[float] = None
    prev_logged_step: Optional[int] = None

    training_start_time = time.time()
    epoch_times: List[float] = []

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_albedo = 0.0
        running_roughness = 0.0
        running_metallic = 0.0
        running_material_iou = 0.0
        running_material_ce = 0.0
        running_material_miou = 0.0
        seen = 0

        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for step_idx, batch in enumerate(pbar, start=1):
            x = batch["input"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            pred_pbr, pred_material = _extract_model_outputs(out)
            if args.task_mode == "pbr":
                y_albedo = batch["albedo"].to(device)
                y_roughness = batch["roughness"].to(device)
                y_metallic = batch["metallic"].to(device)
                y_mask = batch["mask"].to(device)
                if pred_pbr is None:
                    raise RuntimeError("Model did not return PBR prediction in task_mode='pbr'.")

                loss, loss_albedo, loss_roughness, loss_metallic = compute_pbr_loss(
                    pred_pbr,
                    y_albedo,
                    y_roughness,
                    y_metallic,
                    y_mask,
                    albedo_weight=args.albedo_weight,
                    roughness_weight=args.roughness_weight,
                    metallic_weight=args.metallic_weight,
                )
            else:
                if pred_material is None:
                    if pred_pbr is not None and pred_pbr.shape[1] > 1:
                        pred_material = pred_pbr
                    else:
                        raise RuntimeError("Model did not return material logits in task_mode='material'.")
                y_mat = batch["material_id"].to(device).long()
                has_mat = batch["has_material_label"].to(device).view(-1, 1, 1)
                valid_mask = (has_mat > 0.5).expand_as(y_mat)

                iou_loss = _material_iou_loss(
                    pred_material,
                    y_mat,
                    valid_mask=valid_mask,
                    ignore_background=args.material_ignore_background,
                )
                ce_loss = _material_cross_entropy_loss(
                    pred_material,
                    y_mat,
                    valid_mask=valid_mask,
                    ignore_background=args.material_ignore_background,
                )
                miou = _hard_miou(
                    pred_material,
                    y_mat,
                    valid_mask=valid_mask,
                    ignore_background=args.material_ignore_background,
                )
                loss = args.material_weight * iou_loss + args.material_ce_weight * ce_loss
                loss_albedo = torch.tensor(0.0, device=device)
                loss_roughness = torch.tensor(0.0, device=device)
                loss_metallic = torch.tensor(0.0, device=device)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            if args.task_mode == "pbr":
                running_albedo += float(loss_albedo.item()) * bs
                running_roughness += float(loss_roughness.item()) * bs
                running_metallic += float(loss_metallic.item()) * bs
            else:
                running_material_iou += float(iou_loss.item()) * bs
                running_material_ce += float(ce_loss.item()) * bs
                running_material_miou += float(miou.item()) * bs
            seen += bs

            if args.task_mode == "pbr":
                pbar.set_postfix(
                    train_loss=f"{running_loss / max(seen, 1):.4f}",
                    albedo=f"{running_albedo / max(seen, 1):.4f}",
                    rough=f"{running_roughness / max(seen, 1):.4f}",
                    metal=f"{running_metallic / max(seen, 1):.4f}",
                )
            else:
                pbar.set_postfix(
                    train_loss=f"{running_loss / max(seen, 1):.4f}",
                    iou_loss=f"{running_material_iou / max(seen, 1):.4f}",
                    ce_loss=f"{running_material_ce / max(seen, 1):.4f}",
                    miou=f"{running_material_miou / max(seen, 1):.4f}",
                )

            global_step += 1
            if wb is not None and args.log_every_steps > 0 and (global_step % args.log_every_steps == 0):
                current_train_loss = running_loss / max(seen, 1)
                if prev_logged_train_loss is None or prev_logged_step is None:
                    loss_gradient = 0.0
                else:
                    loss_gradient = (current_train_loss - prev_logged_train_loss) / max(global_step - prev_logged_step, 1)

                epoch_progress = (epoch - 1) + (step_idx / max(len(train_loader), 1))
                payload = {
                    "epoch": epoch_progress,
                    "step": global_step,
                    "train/loss": current_train_loss,
                    "train/loss_gradient": loss_gradient,
                }
                if args.task_mode == "pbr":
                    payload["train/loss_albedo"] = running_albedo / max(seen, 1)
                    payload["train/loss_roughness"] = running_roughness / max(seen, 1)
                    payload["train/loss_metallic"] = running_metallic / max(seen, 1)
                else:
                    payload["train/loss_material_iou"] = running_material_iou / max(seen, 1)
                    payload["train/loss_material_ce"] = running_material_ce / max(seen, 1)
                    payload["train/miou_material"] = running_material_miou / max(seen, 1)
                wb.log(payload, step=global_step)
                prev_logged_train_loss = current_train_loss
                prev_logged_step = global_step

        scheduler.step()

        if args.task_mode == "pbr":
            train_metrics = {
                "loss": running_loss / max(seen, 1),
                "loss_albedo": running_albedo / max(seen, 1),
                "loss_roughness": running_roughness / max(seen, 1),
                "loss_metallic": running_metallic / max(seen, 1),
            }
        else:
            train_metrics = {
                "loss": running_loss / max(seen, 1),
                "loss_material_iou": running_material_iou / max(seen, 1),
                "loss_material_ce": running_material_ce / max(seen, 1),
                "miou_material": running_material_miou / max(seen, 1),
            }

        should_validate = (epoch % args.validate_every == 0) or (epoch == args.epochs)

        if should_validate:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                task_mode=args.task_mode,
                albedo_weight=args.albedo_weight,
                roughness_weight=args.roughness_weight,
                metallic_weight=args.metallic_weight,
                material_weight=args.material_weight,
                material_ce_weight=args.material_ce_weight,
                material_ignore_background=args.material_ignore_background,
                lpips_model=lpips_model,
            )

            if args.task_mode == "pbr":
                print(
                    f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                    f"train={train_metrics['loss']:.5f} "
                    f"val={val_metrics['loss']:.5f} "
                    "val_ssim(a/r/m)="
                    f"({val_metrics['ssim_albedo']:.4f}/{val_metrics['ssim_roughness']:.4f}/{val_metrics['ssim_metallic']:.4f})"
                )
                val_panel_target = len(val_vis_ids) if val_vis_ids is not None else args.num_vis_samples
                val_panels = collect_visual_panels(
                    model,
                    val_loader,
                    device,
                    val_panel_target,
                    lpips_model=lpips_model,
                    selected_ids=val_vis_ids,
                )
                _save_visual_panels(vis_dir, "val", epoch, val_panels)
            else:
                print(
                    f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                    f"train={train_metrics['loss']:.5f} "
                    f"val={val_metrics['loss']:.5f} "
                    f"val_mIoU={val_metrics['miou_material']:.4f}"
                )
                val_panels = collect_material_visual_panels(
                    model,
                    val_loader,
                    device,
                    len(val_vis_ids) if val_vis_ids is not None else args.num_vis_samples,
                    material_palette=material_palette,
                    ignore_background=args.material_ignore_background,
                    selected_ids=val_vis_ids,
                )
                _save_visual_panels(vis_dir, "val", epoch, val_panels)

            payload = _build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=None,
                best_val=best_val,
            )

            ckpt_last = ckpt_dir / f"{args.model_name}_last.pth"
            torch.save(payload, ckpt_last)

            ckpt_epoch = ckpt_dir / f"{args.model_name}_epoch_{epoch:03d}.pth"
            torch.save(payload, ckpt_epoch)

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                payload_best = _build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    args=args,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    test_metrics=None,
                    best_val=best_val,
                )
                ckpt_best = ckpt_dir / f"{args.model_name}_best.pth"
                torch.save(payload_best, ckpt_best)

            if wb is not None:
                log_payload = {
                    "epoch": float(epoch),
                    "step": global_step,
                    "train/loss": train_metrics["loss"],
                }
                if args.task_mode == "pbr":
                    log_payload["train/loss_albedo"] = train_metrics["loss_albedo"]
                    log_payload["train/loss_roughness"] = train_metrics["loss_roughness"]
                    log_payload["train/loss_metallic"] = train_metrics["loss_metallic"]
                else:
                    log_payload["train/loss_material_iou"] = train_metrics["loss_material_iou"]
                    log_payload["train/loss_material_ce"] = train_metrics["loss_material_ce"]
                    log_payload["train/miou_material"] = train_metrics["miou_material"]

                for k, v in val_metrics.items():
                    log_payload[f"val/{k}"] = v

                for i, (sample_id, panel) in enumerate(val_panels):
                    log_payload[f"val/vis_{i:02d}"] = wb.Image(panel, caption=sample_id)

                wb.log(log_payload, step=global_step)
        else:
            print(f"[Epoch {epoch:03d}/{args.epochs:03d}] train={train_metrics['loss']:.5f} (skip validation)")
            if wb is not None:
                payload = {
                    "epoch": float(epoch),
                    "step": global_step,
                    "train/loss": train_metrics["loss"],
                }
                if args.task_mode == "pbr":
                    payload["train/loss_albedo"] = train_metrics["loss_albedo"]
                    payload["train/loss_roughness"] = train_metrics["loss_roughness"]
                    payload["train/loss_metallic"] = train_metrics["loss_metallic"]
                else:
                    payload["train/loss_material_iou"] = train_metrics["loss_material_iou"]
                    payload["train/loss_material_ce"] = train_metrics["loss_material_ce"]
                    payload["train/miou_material"] = train_metrics["miou_material"]
                wb.log(payload, step=global_step)

        epoch_time_sec = time.time() - epoch_start_time
        epoch_times.append(epoch_time_sec)
        print(f"Epoch {epoch:03d} time: {epoch_time_sec:.2f} sec")

    total_training_time = time.time() - training_start_time
    mean_epoch_time = total_training_time / max(len(epoch_times), 1)
    print(f"Total training time: {total_training_time:.2f} sec")
    print(f"Mean time per epoch: {mean_epoch_time:.2f} sec")

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    if args.validate_every <= 0:
        raise ValueError("validate_every must be >= 1")
    if args.log_every_steps <= 0:
        raise ValueError("log_every_steps must be >= 1")
    train(args)
