from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_cpu_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        x = x[0]
    return x.detach().cpu()


def denormalize_input(x: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    x = _to_cpu_tensor(x)
    if normalized:
        x = x * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0.0, 1.0)


def to_pil_rgb(x: torch.Tensor) -> Image.Image:
    x = _to_cpu_tensor(x).clamp(0.0, 1.0)
    return T.ToPILImage()(x)


def to_pil_gray(x: torch.Tensor) -> Image.Image:
    x = _to_cpu_tensor(x)
    if x.ndim == 3:
        x = x[0:1]
    x = x.clamp(0.0, 1.0)
    return T.ToPILImage()(x)


def _abs_diff(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (_to_cpu_tensor(pred) - _to_cpu_tensor(gt)).abs().clamp(0.0, 1.0)


def _prepare_mask(mask: torch.Tensor) -> torch.Tensor:
    m = _to_cpu_tensor(mask)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim == 3 and m.shape[0] > 1:
        m = m[0:1]
    return (m > 0.5).float()


def _composite_white(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _prepare_mask(mask)
    inv = 1.0 - m
    t = _to_cpu_tensor(x).clamp(0.0, 1.0)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim == 3 and t.shape[0] == 1:
        return t * m + inv
    if t.ndim == 3 and t.shape[0] == 3:
        return t * m.repeat(3, 1, 1) + inv.repeat(3, 1, 1)
    return t


def _composite_constant(x: torch.Tensor, mask: torch.Tensor, bg_value: float) -> torch.Tensor:
    m = _prepare_mask(mask)
    inv = 1.0 - m
    t = _to_cpu_tensor(x).clamp(0.0, 1.0)
    bg = float(np.clip(bg_value, 0.0, 1.0))
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim == 3 and t.shape[0] == 1:
        return t * m + inv * bg
    if t.ndim == 3 and t.shape[0] == 3:
        m3 = m.repeat(3, 1, 1)
        return t * m3 + (1.0 - m3) * bg
    return t


def _to_np_rgb(x: torch.Tensor) -> np.ndarray:
    t = _to_cpu_tensor(x).clamp(0.0, 1.0)
    if t.ndim == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t.permute(1, 2, 0).numpy()


def _to_np_gray(x: torch.Tensor) -> np.ndarray:
    t = _to_cpu_tensor(x).clamp(0.0, 1.0)
    if t.ndim == 3:
        t = t[0]
    return t.numpy()


def _fmt_eval_line(name: str, metrics: Optional[Dict[str, float]], include_lpips: bool = False) -> str:
    if not metrics:
        return f"{name}: N/A"

    line = (
        f"{name}: MSE {metrics.get('mse', 0.0):.4f}  "
        f"SSIM {metrics.get('ssim', 0.0):.4f}  "
        f"PSNR {metrics.get('psnr', 0.0):.2f}"
    )
    if include_lpips:
        lp = metrics.get('lpips', metrics.get('lpips_albedo', float('nan')))
        if np.isfinite(lp):
            line += f"  LPIPS {lp:.4f}"
    return line


def _fig_to_pil(fig) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return Image.fromarray(arr)


def make_prediction_panel(
    input_img: torch.Tensor,
    pred_albedo: torch.Tensor,
    gt_albedo: Optional[torch.Tensor],
    pred_roughness: torch.Tensor,
    gt_roughness: Optional[torch.Tensor],
    pred_metallic: Optional[torch.Tensor] = None,
    gt_metallic: Optional[torch.Tensor] = None,
    input_is_normalized: bool = True,
    albedo_metrics: Optional[Dict[str, float]] = None,
    roughness_metrics: Optional[Dict[str, float]] = None,
    metallic_metrics: Optional[Dict[str, float]] = None,
    object_mask: Optional[torch.Tensor] = None,
) -> Image.Image:
    def _load_font(size: int) -> ImageFont.FreeTypeFont:
        try:
            fp = font_manager.findfont("DejaVu Sans")
            return ImageFont.truetype(fp, size=size)
        except Exception:
            return ImageFont.load_default()

    def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _gray_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
        g = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)

    def _paste_np(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
        h, w = img.shape[:2]
        canvas[y:y+h, x:x+w] = img

    def _draw_bottom_center(draw: ImageDraw.ImageDraw, x0: int, w: int, y: int, text: str, font) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x0 + (w - tw) // 2, y - th // 2), text, fill=(0, 0, 0), font=font)

    def _draw_vertical_center(canvas_img: Image.Image, x0: int, w: int, y0: int, h: int, text: str, font) -> None:
        tmp = Image.new("RGB", (max(1, h), max(1, w)), (255, 255, 255))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        d.text(((h - tw) // 2, (w - th) // 2), text, fill=(0, 0, 0), font=font)
        rot = tmp.rotate(90, expand=True)
        canvas_img.paste(rot, (x0, y0))

    input_vis = denormalize_input(input_img, normalized=input_is_normalized)
    if object_mask is not None:
        input_vis = _composite_white(input_vis, object_mask)
    input_np = _to_uint8_rgb(_to_np_rgb(input_vis))
    pred_albedo_np = _to_uint8_rgb(_to_np_rgb(pred_albedo))
    pred_rough_np = _gray_to_rgb_uint8(_to_np_gray(pred_roughness))
    if pred_metallic is not None:
        pred_metal_vis = _to_cpu_tensor(pred_metallic).clamp(0.0, 1.0)
        if object_mask is not None:
            pred_metal_vis = _composite_constant(pred_metal_vis, object_mask, bg_value=0.70)
        pred_metal_np = _gray_to_rgb_uint8(_to_np_gray(pred_metal_vis))
    else:
        pred_metal_np = None

    tile_h, tile_w = pred_albedo_np.shape[:2]
    font_title = _load_font(max(24, tile_h // 8))
    font_metrics = _load_font(max(14, tile_h // 18))
    font_labels = _load_font(max(16, tile_h // 14))
    input_img_pil = Image.fromarray(input_np).resize((tile_w, tile_h), Image.Resampling.BILINEAR)
    input_np = np.asarray(input_img_pil)

    if gt_albedo is not None and gt_roughness is not None:
        if object_mask is not None:
            gt_albedo_vis = _composite_white(gt_albedo, object_mask)
            gt_rough_vis = _composite_white(gt_roughness, object_mask)
            gt_metal_vis = _composite_constant(gt_metallic, object_mask, bg_value=0.70) if gt_metallic is not None else None
        else:
            gt_albedo_vis = _to_cpu_tensor(gt_albedo).clamp(0.0, 1.0)
            gt_rough_vis = _to_cpu_tensor(gt_roughness).clamp(0.0, 1.0)
            gt_metal_vis = _to_cpu_tensor(gt_metallic).clamp(0.0, 1.0) if gt_metallic is not None else None

        diff_albedo = _abs_diff(pred_albedo, gt_albedo_vis)
        diff_rough = _abs_diff(pred_roughness, gt_rough_vis)
        if object_mask is not None:
            diff_albedo = _composite_white(diff_albedo, object_mask)
            diff_rough = _composite_white(diff_rough, object_mask)

        gt_albedo_np = _to_uint8_rgb(_to_np_rgb(gt_albedo_vis))
        diff_albedo_np = _to_uint8_rgb(_to_np_rgb(diff_albedo))
        gt_rough_np = _gray_to_rgb_uint8(_to_np_gray(gt_rough_vis))
        diff_rough_np = _gray_to_rgb_uint8(_to_np_gray(diff_rough))

        rows = [
            ("Albedo", pred_albedo_np, gt_albedo_np, diff_albedo_np),
            ("Roughness", pred_rough_np, gt_rough_np, diff_rough_np),
        ]

        if gt_metal_vis is not None and pred_metallic is not None:
            diff_metal = _abs_diff(pred_metallic, gt_metal_vis)
            if object_mask is not None:
                diff_metal = _composite_constant(diff_metal, object_mask, bg_value=0.70)
            gt_metal_np = _gray_to_rgb_uint8(_to_np_gray(gt_metal_vis))
            diff_metal_np = _gray_to_rgb_uint8(_to_np_gray(diff_metal))
            rows.append(("Metallic", pred_metal_np, gt_metal_np, diff_metal_np))

        n_rows = len(rows)
        row_label_w = max(56, tile_w // 6)
        bottom_h = max(34, tile_h // 7)
        metric_lines = [
            _fmt_eval_line("Albedo", albedo_metrics, include_lpips=True),
            _fmt_eval_line("Roughness", roughness_metrics, include_lpips=False),
        ]
        if n_rows >= 3:
            metric_lines.append(_fmt_eval_line("Metallic", metallic_metrics, include_lpips=False))
        title_text = "Evaluation"
        title_bbox = ImageDraw.Draw(Image.new("RGB", (2, 2))).textbbox((0, 0), title_text, font=font_title)
        title_h_px = max(1, title_bbox[3] - title_bbox[1])
        metric_h_px = max(1, max(
            (ImageDraw.Draw(Image.new("RGB", (2, 2))).textbbox((0, 0), line, font=font_metrics)[3]
             - ImageDraw.Draw(Image.new("RGB", (2, 2))).textbbox((0, 0), line, font=font_metrics)[1])
            for line in metric_lines
        ))
        top_pad = max(10, tile_h // 26)
        title_gap = max(10, tile_h // 22)
        line_gap = max(5, tile_h // 34)
        top_h = max(
            90,
            top_pad + title_h_px + title_gap + len(metric_lines) * metric_h_px + max(0, len(metric_lines) - 1) * line_gap + top_pad,
        )
        grid_h = n_rows * tile_h
        canvas_h = top_h + grid_h + bottom_h
        canvas_w = tile_w + row_label_w + 3 * tile_w

        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        # Left input: vertically centered in grid area.
        input_y0 = top_h + (grid_h - tile_h) // 2
        _paste_np(canvas, input_np, 0, input_y0)

        x_grid0 = tile_w + row_label_w
        for r, (row_name, pred_img, gt_img, diff_img) in enumerate(rows):
            y0 = top_h + r * tile_h
            _paste_np(canvas, pred_img, x_grid0 + 0 * tile_w, y0)
            _paste_np(canvas, gt_img,   x_grid0 + 1 * tile_w, y0)
            _paste_np(canvas, diff_img, x_grid0 + 2 * tile_w, y0)

        panel = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(panel)
        y_cursor = top_pad
        _draw_bottom_center(draw, 0, canvas_w, y_cursor + title_h_px // 2, title_text, font_title)
        y_cursor += title_h_px + title_gap
        for line in metric_lines:
            _draw_bottom_center(draw, 0, canvas_w, y_cursor + metric_h_px // 2, line, font_metrics)
            y_cursor += metric_h_px + line_gap

        for r, (row_name, _, _, _) in enumerate(rows):
            _draw_vertical_center(panel, tile_w, row_label_w, top_h + r * tile_h, tile_h, row_name, font_labels)

        y_input = min(canvas_h - max(10, bottom_h // 3), input_y0 + tile_h + max(14, tile_h // 14))
        _draw_bottom_center(draw, 0, tile_w, y_input, "Input View", font_labels)
        y_text = top_h + grid_h + bottom_h // 2
        _draw_bottom_center(draw, x_grid0 + 0 * tile_w, tile_w, y_text, "Pred", font_labels)
        _draw_bottom_center(draw, x_grid0 + 1 * tile_w, tile_w, y_text, "GT", font_labels)
        _draw_bottom_center(draw, x_grid0 + 2 * tile_w, tile_w, y_text, "Diff", font_labels)
        return panel

    # Inference-only mode: keep same white-background style.
    cols = [("Input View", input_np), ("Pred Albedo", pred_albedo_np), ("Pred Roughness", pred_rough_np)]
    if pred_metal_np is not None:
        cols.append(("Pred Metallic", pred_metal_np))

    bottom_h = max(28, tile_h // 8)
    canvas_h = tile_h + bottom_h
    canvas_w = len(cols) * tile_w
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    for i, (_, img) in enumerate(cols):
        _paste_np(canvas, img, i * tile_w, 0)

    panel = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(panel)
    y_text = tile_h + bottom_h // 2
    for i, (name, _) in enumerate(cols):
        _draw_bottom_center(draw, i * tile_w, tile_w, y_text, name, font_labels)
    return panel


def save_inference_outputs(
    output_dir: Path,
    pred_albedo: torch.Tensor,
    pred_roughness: torch.Tensor,
    pred_metallic: Optional[torch.Tensor] = None,
    input_img: Optional[torch.Tensor] = None,
    gt_albedo: Optional[torch.Tensor] = None,
    gt_roughness: Optional[torch.Tensor] = None,
    gt_metallic: Optional[torch.Tensor] = None,
    input_is_normalized: bool = True,
    pred_mask: Optional[torch.Tensor] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    albedo = _to_cpu_tensor(pred_albedo)[:3].clamp(0.0, 1.0)
    rough = _to_cpu_tensor(pred_roughness)
    if rough.ndim == 3:
        rough = rough[0:1]
    rough = rough.clamp(0.0, 1.0)
    if pred_metallic is not None:
        metal = _to_cpu_tensor(pred_metallic)
        if metal.ndim == 3:
            metal = metal[0:1]
        metal = metal.clamp(0.0, 1.0)
    else:
        metal = None

    if pred_mask is not None:
        mask = _prepare_mask(pred_mask)
        inv = 1.0 - mask
        albedo = albedo * mask.repeat(3, 1, 1) + inv.repeat(3, 1, 1)
        rough = rough * mask + inv
        if metal is not None:
            metal = metal * mask + inv

    to_pil_rgb(albedo).save(output_dir / "pred_albedo.png")
    to_pil_gray(rough).save(output_dir / "pred_roughness.png")
    if metal is not None:
        to_pil_gray(metal).save(output_dir / "pred_metallic.png")

    if gt_albedo is not None:
        to_pil_rgb(_abs_diff(albedo, gt_albedo)).save(output_dir / "diff_albedo.png")
    if gt_roughness is not None:
        to_pil_gray(_abs_diff(rough, gt_roughness)).save(output_dir / "diff_roughness.png")
    if gt_metallic is not None and metal is not None:
        to_pil_gray(_abs_diff(metal, gt_metallic)).save(output_dir / "diff_metallic.png")

    if input_img is not None:
        panel = make_prediction_panel(
            input_img=input_img,
            pred_albedo=albedo,
            gt_albedo=gt_albedo,
            pred_roughness=rough,
            gt_roughness=gt_roughness,
            pred_metallic=metal,
            gt_metallic=gt_metallic,
            input_is_normalized=input_is_normalized,
            object_mask=pred_mask,
        )
        panel.save(output_dir / "panel.png")
