import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from resnet50.model import ResNet50UNet
from swin_t.model import SwinTUNet
from utils import save_inference_outputs


DEFAULTS: Dict[str, Any] = {
    "model_name": None,
    "checkpoint": None,
    "input": None,
    "output_dir": "./pred_example",
    "mask": None,
    "image_size": 224,
    "task_mode": "pbr",
    "material_num_classes": None,
    "log_path": None,
    "warmup_steps": 1,
    "repeat_steps": 1,
}


_MATERIAL_COLORS = [
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

    parser = argparse.ArgumentParser(description="Run single-image inference for PBR maps or material segmentation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config. CLI args override config values.")

    parser.add_argument("--model_name", type=str, required=merged["model_name"] is None, default=merged["model_name"], choices=["resnet50_unet", "swin_t_unet"])
    parser.add_argument("--checkpoint", type=str, required=merged["checkpoint"] is None, default=merged["checkpoint"])
    parser.add_argument("--input", type=str, required=merged["input"] is None, help="Path to RGB image", default=merged["input"])
    parser.add_argument("--output_dir", type=str, default=merged["output_dir"])
    parser.add_argument("--mask", type=str, default=merged["mask"], help="Optional object mask path (non-zero is foreground).")
    parser.add_argument("--image_size", type=int, default=merged["image_size"])
    parser.add_argument("--task_mode", type=str, default=merged["task_mode"], choices=["pbr", "material"])
    parser.add_argument(
        "--material_num_classes",
        type=int,
        default=merged["material_num_classes"],
        help="Required for task_mode=material. Must match the trained checkpoint head.",
    )
    parser.add_argument("--log_path", type=str, default=merged["log_path"], help="Optional inference log file path. Relative paths are under output_dir.")
    parser.add_argument("--warmup_steps", type=int, default=merged["warmup_steps"])
    parser.add_argument("--repeat_steps", type=int, default=merged["repeat_steps"])
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


def build_model(
    model_name: str,
    task_mode: str,
    image_size: int = 224,
    material_num_classes: int = 0,
):
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


def _extract_task_output(model_output: torch.Tensor, task_mode: str) -> torch.Tensor:
    if isinstance(model_output, dict):
        key = "pbr" if task_mode == "pbr" else "material_logits"
        out = model_output.get(key)
        if out is None:
            raise RuntimeError(f"Model output dict does not contain '{key}'.")
        return out
    if torch.is_tensor(model_output):
        return model_output
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def _format_flops(flops: Optional[float]) -> str:
    if flops is None:
        return "unavailable"
    return f"{flops / 1e9:.3f} GFLOPs"


def _estimate_model_flops(model: torch.nn.Module, image_size: int, device: torch.device) -> Optional[float]:
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


def _build_material_palette(num_classes: int) -> np.ndarray:
    colors = []
    for cid in range(num_classes):
        if cid < len(_MATERIAL_COLORS):
            colors.append(_MATERIAL_COLORS[cid])
        else:
            colors.append(((37 * cid + 79) % 256, (97 * cid + 31) % 256, (17 * cid + 191) % 256))
    return np.asarray(colors, dtype=np.uint8)


def _denormalize_input(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (x.detach().cpu() * std + mean).clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _save_material_outputs(
    output_dir: Path,
    material_logits: torch.Tensor,
    input_img: torch.Tensor,
    num_classes: int,
    pred_mask: Optional[torch.Tensor] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_ids = material_logits.argmax(dim=0).detach().cpu().long()
    if pred_mask is not None:
        mask = pred_mask.detach().cpu()
        if mask.ndim == 3:
            mask = mask[0]
        pred_ids = pred_ids.masked_fill(mask <= 0.5, 0)

    pred_np = pred_ids.numpy().astype(np.uint8)
    palette = _build_material_palette(num_classes)
    color_np = palette[np.clip(pred_np, 0, num_classes - 1)]

    Image.fromarray(pred_np, mode="L").save(output_dir / "pred_material_ids.png")
    Image.fromarray(color_np, mode="RGB").save(output_dir / "pred_material_color.png")

    input_np = _denormalize_input(input_img)
    panel_np = np.concatenate([input_np, color_np], axis=1)
    panel = Image.fromarray(panel_np, mode="RGB")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), "input", fill=(255, 255, 255))
    draw.text((input_np.shape[1] + 8, 8), "pred material", fill=(255, 255, 255))
    panel.save(output_dir / "panel.png")


def main():
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if args.repeat_steps <= 0:
        raise ValueError("repeat_steps must be >= 1")
    if args.task_mode == "material" and args.material_num_classes is None:
        raise ValueError("task_mode=material requires material_num_classes.")

    total_start = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_log_path = _setup_log_file(output_dir, args.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.config:
        print(f"Loaded config: {args.config}")
    if resolved_log_path is not None:
        print(f"Log path: {resolved_log_path}")

    pbar = tqdm(total=5, desc="Inference", dynamic_ncols=True)

    material_num_classes = int(args.material_num_classes) if args.material_num_classes is not None else 0
    model = build_model(
        args.model_name,
        task_mode=args.task_mode,
        image_size=args.image_size,
        material_num_classes=material_num_classes,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    est_flops = _estimate_model_flops(model, args.image_size, device)
    print(f"Task mode: {args.task_mode}")
    if args.task_mode == "material":
        print(f"Material classes: {material_num_classes}")
    print(f"Model params (total): {num_params_total:,}")
    print(f"Model params (trainable): {num_params_trainable:,}")
    print(f"Model FLOPs (1x3x{args.image_size}x{args.image_size}): {_format_flops(est_flops)}")

    pbar.update(1)
    pbar.set_postfix(step="model_loaded")

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(args.input).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    mask_tensor = None
    if args.mask is not None:
        mask_img = Image.open(args.mask).convert("L")
        mask_transform = T.Compose(
            [
                T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )
        mask_tensor = (mask_transform(mask_img) > 0.5).float()

    pbar.update(1)
    pbar.set_postfix(step="input_prepared")

    with torch.no_grad():
        for _ in range(args.warmup_steps):
            _ = _extract_task_output(model(x), args.task_mode)

        times_ms = []
        pred = None
        for _ in range(args.repeat_steps):
            _maybe_sync(device)
            t0 = time.perf_counter()
            pred = _extract_task_output(model(x), args.task_mode)
            _maybe_sync(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    assert pred is not None
    pred = pred[0].cpu()

    pbar.update(1)
    pbar.set_postfix(step="predicted")

    if args.task_mode == "pbr":
        pred_albedo = pred[:3]
        pred_roughness = pred[3:4]
        pred_metallic = pred[4:5]

        save_inference_outputs(
            output_dir=output_dir,
            pred_albedo=pred_albedo,
            pred_roughness=pred_roughness,
            pred_metallic=pred_metallic,
            input_img=x[0].cpu(),
            input_is_normalized=True,
            pred_mask=mask_tensor,
        )
    else:
        _save_material_outputs(
            output_dir=output_dir,
            material_logits=pred,
            input_img=x[0].cpu(),
            num_classes=material_num_classes,
            pred_mask=mask_tensor,
        )

    pbar.update(1)
    pbar.set_postfix(step="saved")

    mean_ms = sum(times_ms) / max(len(times_ms), 1)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    print(f"Inference time (mean over {len(times_ms)} runs): {mean_ms:.2f} ms")
    print(f"Inference time (min/max): {min_ms:.2f} / {max_ms:.2f} ms")

    total_elapsed = time.time() - total_start
    print(f"Total script time: {total_elapsed:.2f} sec")

    pbar.update(1)
    pbar.set_postfix(step="done")
    pbar.close()

    if args.task_mode == "pbr":
        print(f"Saved: {output_dir / 'pred_albedo.png'}")
        print(f"Saved: {output_dir / 'pred_roughness.png'}")
        print(f"Saved: {output_dir / 'pred_metallic.png'}")
    else:
        print(f"Saved: {output_dir / 'pred_material_ids.png'}")
        print(f"Saved: {output_dir / 'pred_material_color.png'}")
    print(f"Saved: {output_dir / 'panel.png'}")


if __name__ == "__main__":
    main()
