import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
import torch
import torch.nn as nn
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
    "split_file": None,
    "split": "test",
    "model_name": "resnet50_unet",
    "checkpoint": None,
    "image_size": 224,
    "batch_size": 4,
    "num_workers": 4,
    "albedo_weight": 1.0,
    "roughness_weight": 1.0,
    "metallic_weight": 1.0,
    "material_weight": 1.0,
    "material_ce_weight": 0.0,
    "task_mode": "pbr",
    "material_num_classes": None,
    "material_ignore_background": True,
    "material_map_file": None,
    "seed": 42,
    "output_dir": "./eval_outputs",
    "log_path": None,
    "save_visualizations": False,
    "num_vis_samples": 8,
    "vis_seed": 42,
    "vis_ids_file": None,
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
    (30, 30, 30),
    (230, 85, 13),
    (49, 130, 189),
    (227, 26, 28),
    (106, 61, 154),
    (255, 127, 0),
    (177, 89, 40),
    (51, 160, 44),
]


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


def _build_parser(config_defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    merged = dict(DEFAULTS)
    merged.update(config_defaults)

    parser = argparse.ArgumentParser(description="Evaluate inverse-rendering checkpoints on dataset split")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config. CLI args override config values.")

    parser.add_argument("--dataset_root", type=str, required=merged["dataset_root"] in (None, ""), default=merged["dataset_root"])
    parser.add_argument("--split_file", type=str, default=merged["split_file"])
    parser.add_argument("--split", type=str, default=merged["split"], choices=["train", "val", "test"])

    parser.add_argument("--model_name", type=str, default=merged["model_name"], choices=["resnet50_unet", "swin_t_unet"])
    parser.add_argument("--checkpoint", type=str, required=merged["checkpoint"] in (None, ""), default=merged["checkpoint"])
    parser.add_argument("--image_size", type=int, default=merged["image_size"])
    parser.add_argument("--batch_size", type=int, default=merged["batch_size"])
    parser.add_argument("--num_workers", type=int, default=merged["num_workers"])

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
    parser.add_argument("--material_num_classes", type=int, default=merged["material_num_classes"])
    parser.add_argument(
        "--material_ignore_background",
        action=argparse.BooleanOptionalAction,
        default=bool(merged["material_ignore_background"]),
    )
    parser.add_argument("--material_map_file", type=str, default=merged["material_map_file"])
    parser.add_argument("--seed", type=int, default=merged["seed"])

    parser.add_argument("--output_dir", type=str, default=merged["output_dir"])
    parser.add_argument("--log_path", type=str, default=merged["log_path"], help="Optional evaluation log file path. Relative paths are under output_dir.")

    parser.add_argument(
        "--save_visualizations",
        action=argparse.BooleanOptionalAction,
        default=bool(merged["save_visualizations"]),
        help="Save visualization panels for selected samples.",
    )
    parser.add_argument("--num_vis_samples", type=int, default=merged["num_vis_samples"])
    parser.add_argument("--vis_seed", type=int, default=merged["vis_seed"])
    parser.add_argument("--vis_ids_file", type=str, default=merged["vis_ids_file"], help="Optional YAML file that fixes visualization sample ids.")
    return parser


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args()

    config_defaults: Dict[str, Any] = {}
    if pre_args.config:
        config_defaults = _load_config(pre_args.config)

    parser = _build_parser(config_defaults)
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    task_mode: str,
    image_size: int = 224,
    material_num_classes: int = 0,
) -> nn.Module:
    enable_pbr_head = task_mode == "pbr"
    num_material_classes = material_num_classes if task_mode == "material" else 0
    if model_name == "resnet50_unet":
        return ResNet50UNet(
            out_channels=5,
            pretrained=False,
            num_material_classes=num_material_classes,
            enable_pbr_head=enable_pbr_head,
        )
    if model_name == "swin_t_unet":
        return SwinTUNet(
            out_channels=5,
            pretrained=False,
            image_size=image_size,
            num_material_classes=num_material_classes,
            enable_pbr_head=enable_pbr_head,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _format_flops(flops: Optional[float]) -> str:
    if flops is None:
        return "unavailable"
    return f"{flops / 1e9:.3f} GFLOPs"


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


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _setup_log_file(output_dir: Path, log_path: Optional[str]) -> Optional[str]:
    if log_path is None or str(log_path).strip() == "":
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = str(log_path).replace("{ts}", ts)
    p = Path(raw)
    if not p.is_absolute():
        p = output_dir / p
    p.parent.mkdir(parents=True, exist_ok=True)

    log_f = p.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_f)
    sys.stderr = _Tee(sys.stderr, log_f)
    return str(p)


def _extract_model_outputs(model_output) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if isinstance(model_output, dict):
        return model_output.get("pbr"), model_output.get("material_logits")
    if torch.is_tensor(model_output):
        return model_output, None
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def _load_material_segmentation_map(path: Optional[str]) -> Dict[str, int]:
    if path is None or str(path).strip() == "":
        return dict(MATERIAL_SEGMENTATION_MAP)
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"material_map_file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"material_map_file must be dict {{name:id}}, got {type(payload)}")
    if "material_segmentation_map" in payload:
        payload = payload["material_segmentation_map"]
        if not isinstance(payload, dict):
            raise ValueError("material_segmentation_map must be dict {name:id}")
    return {str(k): int(v) for k, v in payload.items()}


def _build_material_palette(num_classes: int) -> np.ndarray:
    colors: List[Tuple[int, int, int]] = []
    for cid in range(num_classes):
        if cid < len(_MATERIAL_COLORS):
            colors.append(_MATERIAL_COLORS[cid])
        else:
            colors.append(((37 * cid + 79) % 256, (97 * cid + 31) % 256, (17 * cid + 191) % 256))
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
    return palette[clipped.reshape(-1)].reshape(h, w, 3)


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


def _make_material_panel(
    input_tensor: torch.Tensor,
    pred_ids: torch.Tensor,
    gt_ids: torch.Tensor,
    sample_miou: float,
    palette: np.ndarray,
) -> Image.Image:
    inp = _input_to_uint8(input_tensor)
    pred_col = _colorize_material_ids(pred_ids.detach().cpu().numpy().astype(np.int64), palette)
    gt_col = _colorize_material_ids(gt_ids.detach().cpu().numpy().astype(np.int64), palette)
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


def compute_pbr_loss(
    pred: torch.Tensor,
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
    pred = torch.sigmoid(pred)
    pred_albedo = pred[:, :3, :, :]
    pred_roughness = pred[:, 3:4, :, :]
    pred_metallic = pred[:, 4:5, :, :]

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


def _load_vis_ids(vis_ids_file: Path) -> List[str]:
    if not vis_ids_file.exists():
        raise FileNotFoundError(f"vis_ids_file not found: {vis_ids_file}")
    with vis_ids_file.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if isinstance(payload, list):
        return [str(x).strip().strip("/") for x in payload]
    if isinstance(payload, dict) and "sample_ids" in payload and isinstance(payload["sample_ids"], list):
        return [str(x).strip().strip("/") for x in payload["sample_ids"]]
    raise ValueError(f"Unsupported vis_ids_file format: {vis_ids_file}")


def _select_visual_ids(dataset: MatPredictDataset, num_vis_samples: int, vis_seed: int, vis_ids_file: Optional[str]) -> List[str]:
    all_ids = sorted([s.sample_id for s in dataset.samples])

    if vis_ids_file is not None and str(vis_ids_file).strip() != "":
        req_ids = _load_vis_ids(Path(vis_ids_file))
        id_set = set(all_ids)
        missing = [sid for sid in req_ids if sid not in id_set]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"{len(missing)} vis ids not found in dataset split. First few: {preview}")
        return req_ids[:num_vis_samples] if num_vis_samples > 0 else req_ids

    if num_vis_samples <= 0:
        return []

    k = min(num_vis_samples, len(all_ids))
    rng = random.Random(vis_seed)
    return sorted(rng.sample(all_ids, k))


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
) -> Tuple[Dict[str, float], Dict[str, float]]:
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
    forward_times_ms = []
    eval_loop_start = time.time()

    pbar = tqdm(loader, desc="Evaluate", dynamic_ncols=True)
    for batch in pbar:
        x = batch["input"].to(device)
        _maybe_sync(device)
        t0 = time.perf_counter()
        out = model(x)
        _maybe_sync(device)
        t1 = time.perf_counter()
        forward_times_ms.append((t1 - t0) * 1000.0)
        pred_pbr, pred_material = _extract_model_outputs(out)

        bs = x.size(0)
        if task_mode == "pbr":
            y_albedo = batch["albedo"].to(device)
            y_roughness = batch["roughness"].to(device)
            y_metallic = batch["metallic"].to(device)
            y_mask = batch["mask"].to(device)
            if pred_pbr is None:
                raise RuntimeError("Model did not return PBR output for task_mode='pbr'.")

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
                    raise RuntimeError("Model did not return material logits for task_mode='material'.")
            y_mat = batch["material_id"].to(device).long()
            has_mat = batch["has_material_label"].to(device).view(-1, 1, 1)
            valid_mask = (has_mat > 0.5).expand_as(y_mat)
            material_num_classes_eval = int(pred_material.shape[1])

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

            pred_ids = pred_material.argmax(dim=1)
            conf_batch = _confusion_matrix(
                pred_ids=pred_ids,
                gt_ids=y_mat,
                valid_mask=valid_mask,
                num_classes=pred_material.shape[1],
            )
            if total_material_conf is None:
                total_material_conf = conf_batch
            else:
                total_material_conf = total_material_conf + conf_batch

        n += bs
        pbar.set_postfix(loss=f"{total_loss / max(n, 1):.4f}")

    total_eval_time_sec = time.time() - eval_loop_start
    total_forward_ms = sum(forward_times_ms)

    if task_mode == "pbr":
        metrics = {
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
    else:
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
        metrics = {
            "loss": total_loss / max(n, 1),
            "loss_material_iou": total_material_iou_loss / max(total_material_count, 1),
            "loss_material_ce": total_material_ce_loss / max(total_material_count, 1),
            **mat_metrics,
        }

    timing = {
        "num_samples": n,
        "num_batches": len(forward_times_ms),
        "mean_forward_ms_per_batch": total_forward_ms / max(len(forward_times_ms), 1),
        "mean_forward_ms_per_sample": total_forward_ms / max(n, 1),
        "total_forward_time_sec": total_forward_ms / 1000.0,
        "total_eval_loop_time_sec": total_eval_time_sec,
        "throughput_samples_per_sec": n / max(total_eval_time_sec, 1e-9),
    }

    return metrics, timing


@torch.no_grad()
def save_selected_visualizations(
    model,
    loader,
    device,
    selected_ids: Sequence[str],
    out_dir: Path,
    task_mode: str,
    material_ignore_background: bool,
    material_palette: Optional[np.ndarray] = None,
    lpips_model=None,
) -> int:
    if not selected_ids:
        return 0

    selected = set(selected_ids)
    saved = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    pbar = tqdm(loader, desc="SaveVis", dynamic_ncols=True)
    for batch in pbar:
        x = batch["input"].to(device)
        ids = batch["id"]

        needs = [i for i, sid in enumerate(ids) if sid in selected]
        if not needs:
            continue

        out = model(x)
        pred_pbr, pred_material = _extract_model_outputs(out)

        for i in needs:
            sid = ids[i]
            if task_mode == "pbr":
                y_albedo = batch["albedo"].to(device)
                y_roughness = batch["roughness"].to(device)
                y_metallic = batch["metallic"].to(device)
                y_mask = batch["mask"].to(device)
                if pred_pbr is None:
                    raise RuntimeError("Expected pbr output for visualization in task_mode='pbr'.")
                pred_albedo = pred_pbr[:, :3, :, :].clamp(0.0, 1.0)
                pred_rough = pred_pbr[:, 3:4, :, :].clamp(0.0, 1.0)
                pred_metal = pred_pbr[:, 4:5, :, :].clamp(0.0, 1.0)

                inv_mask = 1.0 - y_mask
                pred_albedo_vis = pred_albedo * y_mask.repeat(1, 3, 1, 1) + inv_mask.repeat(1, 3, 1, 1)
                pred_rough_vis = pred_rough * y_mask + inv_mask
                pred_metal_vis = pred_metal * y_mask + inv_mask

                m_albedo = _metrics_from_preds(
                    pred_albedo[i : i + 1],
                    y_albedo[i : i + 1],
                    y_mask[i : i + 1].repeat(1, 3, 1, 1),
                )
                m_rough = _metrics_from_preds(
                    pred_rough[i : i + 1],
                    y_roughness[i : i + 1],
                    y_mask[i : i + 1],
                )
                m_metal = _metrics_from_preds(
                    pred_metal[i : i + 1],
                    y_metallic[i : i + 1],
                    y_mask[i : i + 1],
                )
                m_lpips = _masked_lpips_albedo(
                    pred_albedo[i : i + 1],
                    y_albedo[i : i + 1],
                    y_mask[i : i + 1].repeat(1, 3, 1, 1),
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
            else:
                if material_palette is None:
                    raise ValueError("material_palette is required for task_mode='material'.")
                if pred_material is None:
                    if pred_pbr is not None and pred_pbr.shape[1] > 1:
                        pred_material = pred_pbr
                    else:
                        raise RuntimeError("Expected material logits for visualization in task_mode='material'.")
                y_mat = batch["material_id"].to(device).long()
                pred_ids = pred_material.argmax(dim=1)
                conf_i = _confusion_matrix(
                    pred_ids=pred_ids[i],
                    gt_ids=y_mat[i],
                    valid_mask=torch.ones_like(y_mat[i], dtype=torch.bool, device=y_mat.device),
                    num_classes=pred_material.shape[1],
                )
                miou_i = _material_metrics_from_confusion(conf_i, ignore_background=material_ignore_background)["miou_material"]
                panel = _make_material_panel(
                    input_tensor=x[i],
                    pred_ids=pred_ids[i],
                    gt_ids=y_mat[i],
                    sample_miou=miou_i,
                    palette=material_palette,
                )

            safe_id = sid.replace("/", "__")
            panel.save(out_dir / f"{saved:03d}_{safe_id}.png")
            saved += 1

            if saved >= len(selected_ids):
                return saved

    return saved


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_log_path = _setup_log_file(output_dir, args.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if args.config:
        print(f"Loaded config: {args.config}")
    if args.split_file:
        print(f"Using split file: {args.split_file}")
    print(f"Evaluating split: {args.split}")
    if resolved_log_path is not None:
        print(f"Log path: {resolved_log_path}")

    ds = MatPredictDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        image_size=args.image_size,
        seed=args.seed,
        use_imagenet_norm=True,
        split_file=args.split_file,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    inferred_material_num_classes = int(ds.num_material_classes)
    material_num_classes = int(args.material_num_classes) if args.material_num_classes is not None else inferred_material_num_classes
    if args.task_mode == "material" and material_num_classes < 2:
        raise ValueError(
            "task_mode=material requires at least 2 classes (background + 1 material). "
            f"Got material_num_classes={material_num_classes}"
        )
    material_map = _load_material_segmentation_map(args.material_map_file)
    material_palette = _build_material_palette(material_num_classes)

    model = build_model(
        args.model_name,
        task_mode=args.task_mode,
        image_size=args.image_size,
        material_num_classes=material_num_classes,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_flops = _estimate_model_flops(model, args.image_size, device)
    lpips_model = _build_lpips_model(device) if args.task_mode == "pbr" else None

    print(f"Samples: {len(ds)}")
    print(f"Task mode: {args.task_mode}")
    if args.task_mode == "material":
        print(f"Material classes: {material_num_classes} (ignore_background={args.material_ignore_background})")
        print(f"Material loss weights: iou={args.material_weight}, ce={args.material_ce_weight}")
        print(f"Material map: {material_map}")
    print(f"Model params (total): {num_params_total:,}")
    print(f"Model params (trainable): {num_params_trainable:,}")
    print(f"Model FLOPs (1x3x{args.image_size}x{args.image_size}): {_format_flops(est_flops)}")
    if args.task_mode == "pbr":
        print(f"LPIPS(albedo): {'enabled' if lpips_model is not None else 'unavailable (install lpips)'}")

    metrics, timing = evaluate(
        model=model,
        loader=loader,
        device=device,
        task_mode=args.task_mode,
        albedo_weight=args.albedo_weight,
        roughness_weight=args.roughness_weight,
        metallic_weight=args.metallic_weight,
        material_weight=args.material_weight,
        material_ce_weight=args.material_ce_weight,
        material_ignore_background=args.material_ignore_background,
        lpips_model=lpips_model,
    )

    print("Evaluation metrics:")
    if args.task_mode == "pbr":
        keys = [
            "loss",
            "loss_albedo",
            "loss_roughness",
            "loss_metallic",
            "mse_albedo",
            "ssim_albedo",
            "psnr_albedo",
            "lpips_albedo",
            "mse_roughness",
            "ssim_roughness",
            "psnr_roughness",
            "mse_metallic",
            "ssim_metallic",
            "psnr_metallic",
        ]
    else:
        keys = [
            "loss",
            "loss_material_iou",
            "miou_material",
            "fwiou_material",
            "pixel_acc_material",
            "macc_material",
            "mdice_material",
        ]
    for k in keys:
        print(f"  {k}: {metrics[k]:.6f}")

    print("Timing:")
    print(f"  mean forward per batch: {timing['mean_forward_ms_per_batch']:.3f} ms")
    print(f"  mean forward per sample: {timing['mean_forward_ms_per_sample']:.3f} ms")
    print(f"  total forward time: {timing['total_forward_time_sec']:.3f} sec")
    print(f"  total eval loop time: {timing['total_eval_loop_time_sec']:.3f} sec")
    print(f"  throughput: {timing['throughput_samples_per_sec']:.3f} samples/sec")

    vis_ids_used: List[str] = []
    if args.save_visualizations:
        vis_ids_used = _select_visual_ids(
            dataset=ds,
            num_vis_samples=args.num_vis_samples,
            vis_seed=args.vis_seed,
            vis_ids_file=args.vis_ids_file,
        )
        vis_ids_path = output_dir / "vis_sample_ids.yaml"
        with vis_ids_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"sample_ids": vis_ids_used}, f, sort_keys=False)
        print(f"Visualization sample ids saved: {vis_ids_path}")

        vis_dir = output_dir / "visualizations"
        saved_n = save_selected_visualizations(
            model,
            loader,
            device,
            vis_ids_used,
            vis_dir,
            task_mode=args.task_mode,
            material_ignore_background=args.material_ignore_background,
            material_palette=material_palette,
            lpips_model=lpips_model,
        )
        print(f"Saved visualizations: {saved_n} -> {vis_dir}")

    payload = {
        "args": vars(args),
        "model": {
            "num_params_total": num_params_total,
            "num_params_trainable": num_params_trainable,
            "flops": est_flops,
            "flops_human": _format_flops(est_flops),
        },
        "metrics": metrics,
        "timing": timing,
        "vis_sample_ids": vis_ids_used,
    }

    metrics_path = output_dir / f"metrics_{args.split}.yaml"
    with metrics_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
