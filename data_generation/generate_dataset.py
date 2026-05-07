import blenderproc as bproc  # must be first

import argparse
import itertools
import json
import os
import shutil

import bpy
import numpy as np
import yaml

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

MATERIAL_SEGMENTATION_MAP = {
    "background": 0,
    "concrete": 1,
    "fabric": 2,
    "leather": 3,
    "metal": 4,
    "plastic": 5,
    "stone": 6,
    "wood": 7,
}


def parse_blenderproc_args(script_name):
    import sys

    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]

    try:
        script_idx = next(i for i, a in enumerate(argv) if script_name in a)
        return argv[script_idx + 1 :]
    except StopIteration:
        return []


def load_pbr_material(name, folder_path):
    mat = bproc.material.create(name)
    bpy_mat = mat.blender_obj
    bpy_mat.use_nodes = True
    bpy_mat.blend_method = "HASHED"
    bpy_mat.shadow_method = "HASHED"

    nodes = bpy_mat.node_tree.nodes
    links = bpy_mat.node_tree.links
    nodes.clear()

    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Material folder not found: {folder_path}")

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    file_map = {os.path.basename(f).lower(): f for f in files}

    def load_image_node(keywords, is_data=True):
        sorted_items = sorted(file_map.items(), key=lambda x: x[0])
        for kw in keywords:
            for key, path in sorted_items:
                if kw in key:
                    img = bpy.data.images.load(path, check_existing=True)
                    node = nodes.new("ShaderNodeTexImage")
                    node.image = img
                    node.image.colorspace_settings.name = "Non-Color" if is_data else "sRGB"
                    return node
        return None

    tex_base = load_image_node(["basecolor", "albedo", "diffuse"], is_data=False)
    if tex_base:
        links.new(tex_base.outputs["Color"], bsdf_node.inputs["Base Color"])

    tex_rough = load_image_node(["roughness"])
    if tex_rough:
        links.new(tex_rough.outputs["Color"], bsdf_node.inputs["Roughness"])

    tex_metal = load_image_node(["metallic", "metalness"])
    if tex_metal:
        links.new(tex_metal.outputs["Color"], bsdf_node.inputs["Metallic"])

    tex_alpha = load_image_node(["opacity", "alpha"])
    if tex_alpha:
        links.new(tex_alpha.outputs["Color"], bsdf_node.inputs["Alpha"])

    tex_norm = load_image_node(["normal"])
    if tex_norm:
        norm_map = nodes.new("ShaderNodeNormalMap")
        links.new(tex_norm.outputs["Color"], norm_map.inputs["Color"])
        links.new(norm_map.outputs["Normal"], bsdf_node.inputs["Normal"])

    return mat


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_variant_maps(material_candidates, strategy, max_variants):
    keys = list(material_candidates.keys())
    values = [material_candidates[k] for k in keys]

    if not keys:
        return [{}]

    for k, v in material_candidates.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"material_candidates[{k}] must be a non-empty list")

    variants = []
    if strategy == "cartesian":
        for combo in itertools.product(*values):
            variants.append({k: m for k, m in zip(keys, combo)})
    elif strategy == "zip":
        limit = min(len(v) for v in values)
        for i in range(limit):
            variants.append({k: values[idx][i] for idx, k in enumerate(keys)})
    else:
        raise ValueError(f"Unsupported variant strategy: {strategy}")

    if max_variants is not None:
        variants = variants[: int(max_variants)]
    return variants


def normalize_slot_name(name):
    return name.lower().split(".")[0]


def set_material_label_id(material_obj, label_id):
    label_id = int(label_id)
    material_obj.blender_obj["label_id"] = label_id
    material_obj.blender_obj.pass_index = label_id


def assign_materials_manual(objs, manual_cfg, variant_map, materials_root):
    slot_to_key = manual_cfg.get("slot_to_key", {})
    fallback_key = manual_cfg.get("fallback_key")
    if not slot_to_key and fallback_key is None:
        # fallback: use the first key in variant map
        fallback_key = sorted(variant_map.keys())[0]

    label_id_map = build_label_id_map(variant_map)
    material_cache = {}
    for obj in objs:
        mats = obj.get_materials()
        for i, m in enumerate(mats):
            slot_name = normalize_slot_name(m.get_name()) if m else ""
            key = slot_to_key.get(slot_name, fallback_key)
            if key is None:
                continue
            if key not in variant_map:
                raise KeyError(f"Material key '{key}' not found in variant_map keys {list(variant_map.keys())}")
            material_name = variant_map[key]
            cache_key = (key, material_name)
            if cache_key not in material_cache:
                mat_path = os.path.join(materials_root, material_name)
                material_cache[cache_key] = load_pbr_material(f"{key}_{material_name}", mat_path)
                set_material_label_id(material_cache[cache_key], label_id_map.get(key, 0))
            obj.set_material(i, material_cache[cache_key])


def assign_uniform_material(objs, material_obj, label_id=1):
    set_material_label_id(material_obj, label_id)
    for obj in objs:
        mats = obj.get_materials()
        if not mats:
            obj.add_material(material_obj)
            continue
        for idx in range(len(mats)):
            obj.set_material(idx, material_obj)


def collect_world_vertices(objs):
    verts_world = []
    for obj in objs:
        bpy_obj = obj.blender_obj
        if bpy_obj.type != "MESH" or bpy_obj.data is None:
            continue
        mw = np.array(bpy_obj.matrix_world)
        local = np.array([v.co for v in bpy_obj.data.vertices])
        if local.size == 0:
            continue
        local_h = np.hstack([local, np.ones((local.shape[0], 1), dtype=np.float32)])
        world = (mw @ local_h.T).T[:, :3]
        verts_world.append(world)
    if not verts_world:
        return None
    return np.vstack(verts_world)


def build_camera_poses(objs, render_cfg):
    width = int(render_cfg.get("resolution", {}).get("width", 512))
    height = int(render_cfg.get("resolution", {}).get("height", 512))
    bproc.camera.set_resolution(width, height)

    camera_cfg = render_cfg.get("camera", {})
    azimuths = np.deg2rad(np.array(camera_cfg.get("azimuth_deg", [0, 90, 180, 270]), dtype=np.float32))
    elevations = np.deg2rad(np.array(camera_cfg.get("elevation_deg", [0, 30]), dtype=np.float32))
    z_offset = float(camera_cfg.get("z_offset", 0.0))

    radius_cfg = render_cfg.get("radius", {})
    min_radius = float(radius_cfg.get("min", 1.0))
    max_radius = radius_cfg.get("max", None)
    if max_radius in ("", "null"):
        max_radius = None
    if max_radius is not None:
        max_radius = float(max_radius)

    verts = collect_world_vertices(objs)
    if verts is None:
        poi = np.array(camera_cfg.get("poi", [0.0, 0.0, 0.0]), dtype=np.float32)
        radius = min_radius
    else:
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        poi = (bbox_min + bbox_max) / 2.0
        extent = float(np.max(bbox_max - bbox_min))
        radius = max(min_radius, extent * 2.2)
        if max_radius is not None:
            radius = min(radius, max_radius)

    frames_data = []
    frame_id = 0
    for el in elevations:
        for az in azimuths:
            location = [
                radius * np.cos(el) * np.cos(az),
                radius * np.cos(el) * np.sin(az),
                radius * np.sin(el) + z_offset,
            ]
            location = np.array(location) + poi
            rot_mat = bproc.camera.rotation_from_forward_vec(poi - location)
            cam_mat = bproc.math.build_transformation_mat(location, rot_mat)
            bproc.camera.add_camera_pose(cam_mat, frame=frame_id)
            frames_data.append({"frame_id": frame_id, "transform_matrix_c2w": cam_mat})
            frame_id += 1

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(0, len(frames_data) - 1)
    bpy.context.scene.frame_set(0)
    return frames_data


def _compute_poi_extent_from_objs(objs):
    verts = collect_world_vertices(objs)
    if verts is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.0
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    poi = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
    extent = float(np.max(bbox_max - bbox_min))
    extent = max(extent, 0.5)
    return poi, extent


def setup_lighting(objs, frames_data, render_cfg):
    _ = frames_data
    lighting_cfg = render_cfg.get("lighting", {})
    mode = str(lighting_cfg.get("mode", "fixed_three_point")).strip().lower()

    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        amb = float(lighting_cfg.get("ambient_strength", 0.08))
        bg_node.inputs[0].default_value = (amb, amb, amb, 1.0)

    poi, extent = _compute_poi_extent_from_objs(objs)

    # Indoor-like setup: one ceiling area light + three weak spot lights.
    if mode in ("fixed_three_point", "ceiling_area_spots"):
        ceiling = bproc.types.Light()
        ceiling.set_type("AREA")
        ceiling_energy = float(lighting_cfg.get("ceiling_energy", 400.0))
        ceiling.set_energy(ceiling_energy)
        ceiling_h = float(lighting_cfg.get("ceiling_height", max(4.2, extent * 5.2)))
        ceiling.set_location((poi + np.array([0.0, 0.0, ceiling_h], dtype=np.float32)).tolist())
        try:
            ceiling.blender_obj.data.size = float(lighting_cfg.get("ceiling_size", max(1.5, extent * 2.5)))
            if hasattr(ceiling.blender_obj.data, "shape"):
                ceiling.blender_obj.data.shape = "SQUARE"
        except Exception:
            pass

        spot_energy = float(lighting_cfg.get("spot_energy", 150.0))
        spot_radius = float(lighting_cfg.get("spot_radius", max(1.2, extent * 1.5)))
        spot_height = float(lighting_cfg.get("spot_height", max(0.5, extent * 0.7)))
        spot_size_deg = float(lighting_cfg.get("spot_size_deg", 75.0))
        spot_blend = float(lighting_cfg.get("spot_blend", 0.35))

        for az in (0.0, 120.0, 240.0):
            rad = np.deg2rad(az)
            loc = poi + np.array(
                [spot_radius * np.cos(rad), spot_radius * np.sin(rad), spot_height],
                dtype=np.float32,
            )
            spot = bproc.types.Light()
            spot.set_type("SPOT")
            spot.set_energy(spot_energy)
            spot.set_location(loc.tolist())
            try:
                rot = bproc.camera.rotation_from_forward_vec(poi - loc)
                spot.set_rotation_euler(rot)
            except Exception:
                pass
            try:
                spot.blender_obj.data.spot_size = float(np.deg2rad(spot_size_deg))
                spot.blender_obj.data.spot_blend = spot_blend
                spot.blender_obj.data.shadow_soft_size = float(lighting_cfg.get("spot_soft_size", 0.8))
            except Exception:
                pass
        return

    # Fallback: keep previous random 3-point behavior.
    key_light = bproc.types.Light()
    fill_light = bproc.types.Light()
    rim_light = bproc.types.Light()
    key_light.blender_obj.data.shadow_soft_size = 1.0
    fill_light.blender_obj.data.shadow_soft_size = 3.0
    rim_light.blender_obj.data.shadow_soft_size = 1.5

    rng = np.random.default_rng(42)
    for frame in frames_data:
        frame_id = frame["frame_id"]
        c2w = np.array(frame["transform_matrix_c2w"])
        location = c2w[:3, 3]

        key_light.set_energy(float(rng.uniform(300, 800)), frame=frame_id)
        key_loc = location + np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 2)])
        key_light.set_location(key_loc, frame=frame_id)

        fill_light.set_energy(float(rng.uniform(50, 200)), frame=frame_id)
        fill_loc = -location + np.array([0, 0, rng.uniform(1, 2)])
        fill_light.set_location(fill_loc, frame=frame_id)

        rim_light.set_energy(float(rng.uniform(100, 500)), frame=frame_id)
        rim_loc = [rng.uniform(-2, 2), rng.uniform(2, 4), rng.uniform(0.5, 2)]
        rim_light.set_location(rim_loc, frame=frame_id)


def setup_camera_and_light(objs, render_cfg):
    frames_data = build_camera_poses(objs, render_cfg)
    setup_lighting(objs, frames_data, render_cfg)

    # Keep renderer outputs in raw/linear for data channels (AOVs).
    bpy.context.scene.view_settings.view_transform = "Raw"
    bpy.context.scene.view_settings.exposure = 0.0

    print(f"[Camera] total views={len(frames_data)}")
    return frames_data

def _link_socket_or_default(nodes, links, source_input, target_input):
    if source_input.links:
        links.new(source_input.links[0].from_socket, target_input)
        return

    val = source_input.default_value
    if isinstance(val, (float, int)):
        v = float(val)
        target_input.default_value = (v, v, v, 1.0)
    elif hasattr(val, "__len__") and len(val) == 3:
        target_input.default_value = (val[0], val[1], val[2], 1.0)
    else:
        target_input.default_value = val


def _link_value_or_default(nodes, links, source_input, target_input):
    if source_input.links:
        from_sock = source_input.links[0].from_socket
        # Convert color texture output to scalar when writing VALUE AOV.
        if hasattr(from_sock, "type") and from_sock.type == "RGBA":
            rgb2bw = nodes.new("ShaderNodeRGBToBW")
            links.new(from_sock, rgb2bw.inputs["Color"])
            links.new(rgb2bw.outputs["Val"], target_input)
        else:
            links.new(from_sock, target_input)
        return

    val = source_input.default_value
    if isinstance(val, (float, int)):
        target_input.default_value = float(val)
    elif hasattr(val, "__len__") and len(val) >= 1:
        target_input.default_value = float(val[0])
    else:
        target_input.default_value = 0.0


def _list_sorted_files(folder, suffix):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, n)
        for n in os.listdir(folder)
        if n.lower().endswith(suffix.lower())
    )


def write_transforms_json(output_dir, frames_data):
    k = np.array(bproc.camera.get_intrinsics_as_K_matrix(), dtype=np.float64)
    width = float(bpy.context.scene.render.resolution_x)
    fx = float(k[0, 0])
    camera_angle_x = float(2.0 * np.arctan(width / (2.0 * fx))) if fx > 0 else 0.6911112070083618
    frames = []
    for i, f in enumerate(frames_data):
        stem = f"{i:03d}"
        frames.append(
            {
                "envmap": None,
                "rotation": -0.017453292519943295,
                "transform_matrix": np.array(f["transform_matrix_c2w"]).tolist(),
                "file_path": f"images/{stem}",
                "file_path_albedo": f"albedo/{stem}",
                "file_path_normal": f"normal_obj/{stem}",
                "file_path_orm": f"ORM/{stem}",
                "file_path_depth": f"depth/{stem}",
            }
        )
    with open(os.path.join(output_dir, "transforms.json"), "w", encoding="utf-8") as fp:
        json.dump({"frames": frames, "camera_angle_x": camera_angle_x}, fp, indent=2)


# --- Material segmentation labels, channel outputs, and image dedup ---
_CLIP_MATCHER = {"ready": False, "backend": None, "embed": None}


def _material_family_from_name(material_name):
    name = str(material_name or "").strip().lower()
    for fam in ("concrete", "fabric", "leather", "metal", "plastic", "stone", "wood"):
        if name.startswith(f"{fam}_") or name == fam:
            return fam
    return "background"


def build_label_id_map(variant_map):
    label_map = {}
    for key, material_name in (variant_map or {}).items():
        fam = _material_family_from_name(material_name)
        label_map[key] = int(MATERIAL_SEGMENTATION_MAP.get(fam, 0))
    return label_map


def write_material_segmentation_table(output_dir):
    out = {
        "id_to_material_family": {str(v): k for k, v in MATERIAL_SEGMENTATION_MAP.items()},
        "material_family_to_id": dict(MATERIAL_SEGMENTATION_MAP),
    }
    with open(os.path.join(output_dir, "material_segmentation_map.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)


def _read_image_rgb(path):
    if os.path.isfile(path):
        if imageio is not None:
            arr = imageio.imread(path)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] >= 4:
                arr = arr[..., :3]
            return arr.astype(np.float32) / 255.0

        img = bpy.data.images.load(path, check_existing=False)
        try:
            w, h = img.size
            px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
            return px[..., :3]
        finally:
            bpy.data.images.remove(img)
    raise FileNotFoundError(path)


def _simple_ssim(img_a, img_b):
    # Lightweight global SSIM approximation (no external deps).
    gray_a = 0.299 * img_a[..., 0] + 0.587 * img_a[..., 1] + 0.114 * img_a[..., 2]
    gray_b = 0.299 * img_b[..., 0] + 0.587 * img_b[..., 1] + 0.114 * img_b[..., 2]

    mu_a = float(gray_a.mean())
    mu_b = float(gray_b.mean())
    var_a = float(((gray_a - mu_a) ** 2).mean())
    var_b = float(((gray_b - mu_b) ** 2).mean())
    cov_ab = float(((gray_a - mu_a) * (gray_b - mu_b)).mean())

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    denom = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if denom <= 1e-12:
        return 1.0
    return float(((2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)) / denom)


def _read_image_mask(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if imageio is not None:
        arr = imageio.imread(path)
        if arr.ndim == 2:
            return arr > 0
        if arr.ndim == 3 and arr.shape[-1] >= 4:
            return arr[..., 3] > 0
        rgb = arr[..., :3].astype(np.float32)
        return np.any(rgb > 0, axis=-1)

    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        alpha = px[..., 3]
        if np.max(alpha) > 1e-6:
            return alpha > 1e-6
        return np.any(px[..., :3] > 1e-6, axis=-1)
    finally:
        bpy.data.images.remove(img)


def _mask_iou(mask_a, mask_b):
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + 1e-12))


def _init_clip_matcher():
    if _CLIP_MATCHER["ready"]:
        return _CLIP_MATCHER["embed"]

    # Backend 1: open_clip
    try:
        import torch
        import open_clip
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        model.eval()

        def _embed(path):
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                t = preprocess(img).unsqueeze(0)
                feat = model.encode_image(t)
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
            return feat[0].cpu().numpy().astype(np.float32)

        _CLIP_MATCHER.update({"ready": True, "backend": "open_clip", "embed": _embed})
        print("[Dedup] CLIP backend: open_clip")
        return _embed
    except Exception:
        pass

    # Backend 2: transformers CLIP
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        def _embed(path):
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
            return feat[0].cpu().numpy().astype(np.float32)

        _CLIP_MATCHER.update({"ready": True, "backend": "transformers_clip", "embed": _embed})
        print("[Dedup] CLIP backend: transformers")
        return _embed
    except Exception:
        pass

    _CLIP_MATCHER.update({"ready": True, "backend": None, "embed": None})
    print("[Dedup][WARN] CLIP backend unavailable. Fallback to SSIM-only dedup.")
    return None


def _compute_clip_distance(path_a, path_b):
    embed_fn = _init_clip_matcher()
    if embed_fn is None:
        return None
    fa = embed_fn(path_a)
    fb = embed_fn(path_b)
    cos_sim = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))
    return float(1.0 - cos_sim)


def deduplicate_rendered_views(output_dir, frames_data, dedup_cfg):
    cfg = dict(dedup_cfg or {})
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return frames_data

    ssim_dist_th = float(cfg.get("ssim_distance_threshold", 0.05))
    iou_th = float(cfg.get("iou_threshold", 0.90))
    top_k = max(1, int(cfg.get("clip_top_k", 5)))

    channels = {
        "images": ".png",
        "albedo": ".png",
        "depth": ".exr",
        "ORM": ".png",
        "normal_mat": ".png",
        "normal_obj": ".png",
        "label": ".png",
    }

    image_dir = os.path.join(output_dir, "images")
    stems = sorted([os.path.splitext(n)[0] for n in os.listdir(image_dir) if n.lower().endswith(".png")])
    if len(stems) <= 1:
        return frames_data

    embed_fn = _init_clip_matcher()
    clip_available = embed_fn is not None
    emb_cache = {}

    def get_emb(stem):
        if stem in emb_cache:
            return emb_cache[stem]
        emb_cache[stem] = embed_fn(os.path.join(image_dir, f"{stem}.png"))
        return emb_cache[stem]

    keep_stems = [stems[0]]
    drop_stems = []

    for stem in stems[1:]:
        cur_path = os.path.join(image_dir, f"{stem}.png")
        img_cur = _read_image_rgb(cur_path)
        mask_cur = _read_image_mask(cur_path)

        if clip_available:
            cur_emb = get_emb(stem)
            dists = []
            for ks in keep_stems:
                ke = get_emb(ks)
                cos_sim = float(np.dot(cur_emb, ke) / (np.linalg.norm(cur_emb) * np.linalg.norm(ke) + 1e-12))
                dists.append((1.0 - cos_sim, ks))
            dists.sort(key=lambda x: x[0])
            candidate_stems = [ks for _, ks in dists[: min(top_k, len(dists))]]
        else:
            candidate_stems = list(keep_stems)

        is_duplicate = False
        for ks in candidate_stems:
            keep_path = os.path.join(image_dir, f"{ks}.png")
            img_keep = _read_image_rgb(keep_path)
            mask_keep = _read_image_mask(keep_path)

            ssim_val = _simple_ssim(img_keep, img_cur)
            ssim_dist = 1.0 - ssim_val
            iou_val = _mask_iou(mask_keep, mask_cur)

            if (ssim_dist <= ssim_dist_th) or (iou_val >= iou_th):
                is_duplicate = True
                break

        if is_duplicate:
            drop_stems.append(stem)
        else:
            keep_stems.append(stem)

    if not drop_stems:
        print("[Dedup] No near-duplicate views removed.")
        return frames_data

    print(
        f"[Dedup] kept={len(keep_stems)} dropped={len(drop_stems)} "
        f"(strategy=clip_topk_then_ssim_iou, top_k={top_k}, "
        f"ssim_dist_th={ssim_dist_th}, iou_th={iou_th}, clip_available={clip_available})"
    )

    # 1) Remove dropped stems in all channels.
    for stem in drop_stems:
        for ch, ext in channels.items():
            p = os.path.join(output_dir, ch, f"{stem}{ext}")
            if os.path.isfile(p):
                os.remove(p)

    # 2) Renumber kept stems to contiguous 000..N-1 in all channels.
    for new_idx, old_stem in enumerate(keep_stems):
        new_stem = f"{new_idx:03d}"
        if new_stem == old_stem:
            continue
        for ch, ext in channels.items():
            old_p = os.path.join(output_dir, ch, f"{old_stem}{ext}")
            if not os.path.isfile(old_p):
                continue
            tmp_p = os.path.join(output_dir, ch, f"__tmp__{new_stem}{ext}")
            os.rename(old_p, tmp_p)

    # Second pass: tmp -> final names.
    for ch, ext in channels.items():
        ch_dir = os.path.join(output_dir, ch)
        for n in sorted(os.listdir(ch_dir)):
            if n.startswith("__tmp__") and n.endswith(ext):
                final_name = n.replace("__tmp__", "", 1)
                os.rename(os.path.join(ch_dir, n), os.path.join(ch_dir, final_name))

    # Keep frame transforms aligned with kept image indices.
    kept_indices = [int(s) for s in keep_stems]
    return [frames_data[i] for i in kept_indices]


def add_shader_aov_outputs():
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    aov_defs = {
        "basecolor": {"name": "aov_basecolor", "type": "COLOR"},
        "roughness": {"name": "aov_roughness", "type": "VALUE"},
        "metallic": {"name": "aov_metallic", "type": "VALUE"},
        "occlusion": {"name": "aov_occlusion", "type": "VALUE"},
        "opacity": {"name": "aov_opacity", "type": "VALUE"},
        "normal_tex": {"name": "aov_normal_tex", "type": "COLOR"},
        "normal_obj": {"name": "aov_normal_obj", "type": "COLOR"},
    }

    for cfg in aov_defs.values():
        name = cfg["name"]
        if view_layer.aovs.get(name) is None:
            aov = view_layer.aovs.add()
            aov.name = name
            aov.type = cfg["type"]
        else:
            view_layer.aovs[name].type = cfg["type"]

    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue

        for n in [n for n in nodes if n.type == "OUTPUT_AOV"]:
            nodes.remove(n)
        for n in [n for n in nodes if n.type == "VALUE" and n.name.startswith("LABEL_ID_VALUE")]:
            nodes.remove(n)

        for key, cfg in aov_defs.items():
            aov_name = cfg["name"]
            out = nodes.new("ShaderNodeOutputAOV")
            out.name = f"OUT_{aov_name}"
            out.aov_name = aov_name

            if key == "normal_tex":
                if bsdf.inputs["Normal"].links:
                    nm_node = bsdf.inputs["Normal"].links[0].from_node
                    if nm_node.type == "NORMAL_MAP" and nm_node.inputs["Color"].links:
                        links.new(nm_node.inputs["Color"].links[0].from_socket, out.inputs["Color"])
                    else:
                        links.new(bsdf.inputs["Normal"].links[0].from_socket, out.inputs["Color"])
                else:
                    out.inputs["Color"].default_value = (0.5, 0.5, 1.0, 1.0)
            elif key == "normal_obj":
                geom = nodes.new("ShaderNodeNewGeometry")
                scale = nodes.new("ShaderNodeVectorMath")
                scale.operation = "SCALE"
                scale.inputs["Scale"].default_value = 0.5
                bias = nodes.new("ShaderNodeVectorMath")
                bias.operation = "ADD"
                bias.inputs[1].default_value = (0.5, 0.5, 0.5)
                links.new(geom.outputs["True Normal"], scale.inputs[0])
                links.new(scale.outputs["Vector"], bias.inputs[0])
                links.new(bias.outputs["Vector"], out.inputs["Color"])
            elif key == "basecolor":
                _link_socket_or_default(nodes, links, bsdf.inputs["Base Color"], out.inputs["Color"])
            elif key == "roughness":
                _link_value_or_default(nodes, links, bsdf.inputs["Roughness"], out.inputs["Value"])
            elif key == "metallic":
                _link_value_or_default(nodes, links, bsdf.inputs["Metallic"], out.inputs["Value"])
            elif key == "occlusion":
                ao = nodes.new("ShaderNodeAmbientOcclusion")
                ao.inputs["Distance"].default_value = 1.0
                if "AO" in ao.outputs:
                    links.new(ao.outputs["AO"], out.inputs["Value"])
                else:
                    rgb2bw = nodes.new("ShaderNodeRGBToBW")
                    links.new(ao.outputs["Color"], rgb2bw.inputs["Color"])
                    links.new(rgb2bw.outputs["Val"], out.inputs["Value"])
    return aov_defs


def setup_compositor_outputs(output_dir, aov_defs, rgb_exposure=1.0, depth_scale=5000, depth_max_m=50.0):
    _ = (depth_scale, depth_max_m)
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_material_index = True
    try:
        view_layer.update()
    except Exception:
        pass
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    links = tree.links

    tmp_root = os.path.join(output_dir, "_tmp_channels")
    os.makedirs(tmp_root, exist_ok=True)
    rl = tree.nodes.new("CompositorNodeRLayers")

    def _link_linear_to_srgb(in_socket, out_socket):
        try:
            csc = tree.nodes.new("CompositorNodeConvertColorSpace")
            try:
                csc.from_color_space = "Linear"
                csc.to_color_space = "sRGB"
            except Exception:
                csc.from_color_space = "scene_linear"
                csc.to_color_space = "sRGB"
            links.new(in_socket, csc.inputs["Image"])
            links.new(csc.outputs["Image"], out_socket)
            return
        except Exception:
            pass

        gamma = tree.nodes.new("CompositorNodeGamma")
        gamma.inputs["Gamma"].default_value = 1.0 / 2.2
        links.new(in_socket, gamma.inputs["Image"])
        links.new(gamma.outputs["Image"], out_socket)

    rgb_out = tree.nodes.new("CompositorNodeOutputFile")
    rgb_out.base_path = os.path.join(tmp_root, "images")
    rgb_out.format.file_format = "PNG"
    rgb_out.format.color_mode = "RGBA"
    rgb_out.format.color_depth = "8"
    rgb_out.format.compression = 100
    rgb_out.file_slots[0].path = "frame_"

    depth_out = tree.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = os.path.join(tmp_root, "depth")
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_mode = "RGBA"
    depth_out.format.color_depth = "16"
    try:
        depth_out.format.exr_codec = "PIZ"
    except Exception:
        pass
    depth_out.file_slots[0].path = "frame_"

    available = [sock.name for sock in rl.outputs]
    alpha_socket_name = "Alpha" if "Alpha" in available else None
    label_socket_name = next(
        (candidate for candidate in ("IndexMA", "Material Index", "Index Material") if candidate in available),
        None,
    )

    rgb_exposure_node = tree.nodes.new("CompositorNodeExposure")
    rgb_exposure_node.inputs["Exposure"].default_value = float(rgb_exposure)
    links.new(rl.outputs["Image"], rgb_exposure_node.inputs["Image"])
    if alpha_socket_name is not None:
        rgb_set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
        _link_linear_to_srgb(rgb_exposure_node.outputs["Image"], rgb_set_alpha.inputs["Image"])
        links.new(rl.outputs[alpha_socket_name], rgb_set_alpha.inputs["Alpha"])
        links.new(rgb_set_alpha.outputs["Image"], rgb_out.inputs[0])
    else:
        _link_linear_to_srgb(rgb_exposure_node.outputs["Image"], rgb_out.inputs[0])

    depth_socket_name = None
    for candidate in ("Depth", "Z"):
        if candidate in available:
            depth_socket_name = candidate
            break
    if depth_socket_name is not None:
        less_than = tree.nodes.new("CompositorNodeMath")
        less_than.operation = "LESS_THAN"
        less_than.inputs[1].default_value = float(depth_max_m)

        mul_mask = tree.nodes.new("CompositorNodeMath")
        mul_mask.operation = "MULTIPLY"

        min_clip = tree.nodes.new("CompositorNodeMath")
        min_clip.operation = "MINIMUM"
        min_clip.inputs[1].default_value = float(depth_max_m)

        links.new(rl.outputs[depth_socket_name], less_than.inputs[0])
        links.new(rl.outputs[depth_socket_name], mul_mask.inputs[0])
        links.new(less_than.outputs[0], mul_mask.inputs[1])
        links.new(mul_mask.outputs[0], min_clip.inputs[0])

        depth_for_obj = min_clip.outputs[0]
        if alpha_socket_name is not None:
            mul_alpha = tree.nodes.new("CompositorNodeMath")
            mul_alpha.operation = "MULTIPLY"
            links.new(depth_for_obj, mul_alpha.inputs[0])
            links.new(rl.outputs[alpha_socket_name], mul_alpha.inputs[1])
            depth_for_obj = mul_alpha.outputs[0]

        depth_rgb = tree.nodes.new("CompositorNodeCombRGBA")
        links.new(depth_for_obj, depth_rgb.inputs["R"])
        links.new(depth_for_obj, depth_rgb.inputs["G"])
        links.new(depth_for_obj, depth_rgb.inputs["B"])
        if alpha_socket_name is not None:
            links.new(rl.outputs[alpha_socket_name], depth_rgb.inputs["A"])
        else:
            depth_rgb.inputs["A"].default_value = 1.0
        links.new(depth_rgb.outputs["Image"], depth_out.inputs[0])

    socket_map = {}
    for cfg in aov_defs.values():
        aov_name = cfg["name"]
        socket_name = None
        for candidate in (aov_name, f"AOV {aov_name}"):
            if candidate in available:
                socket_name = candidate
                break
        if socket_name is None:
            continue
        socket_map[aov_name] = socket_name

        if aov_name == "aov_basecolor":
            out_srgb = tree.nodes.new("CompositorNodeOutputFile")
            out_srgb.base_path = os.path.join(tmp_root, "albedo")
            out_srgb.format.file_format = "PNG"
            out_srgb.format.color_mode = "RGBA"
            out_srgb.format.color_depth = "8"
            out_srgb.format.compression = 100
            out_srgb.file_slots[0].path = "frame_"
            if alpha_socket_name is not None:
                alb_set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
                _link_linear_to_srgb(rl.outputs[socket_name], alb_set_alpha.inputs["Image"])
                links.new(rl.outputs[alpha_socket_name], alb_set_alpha.inputs["Alpha"])
                links.new(alb_set_alpha.outputs["Image"], out_srgb.inputs[0])
            else:
                _link_linear_to_srgb(rl.outputs[socket_name], out_srgb.inputs[0])
        elif aov_name == "aov_normal_tex":
            out = tree.nodes.new("CompositorNodeOutputFile")
            out.base_path = os.path.join(tmp_root, "normal_mat")
            out.format.file_format = "PNG"
            out.format.color_mode = "RGBA"
            out.format.color_depth = "8"
            out.format.compression = 85
            out.file_slots[0].path = "frame_"
            if alpha_socket_name is not None:
                normal_set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
                links.new(rl.outputs[socket_name], normal_set_alpha.inputs["Image"])
                links.new(rl.outputs[alpha_socket_name], normal_set_alpha.inputs["Alpha"])
                links.new(normal_set_alpha.outputs["Image"], out.inputs[0])
            else:
                links.new(rl.outputs[socket_name], out.inputs[0])
        elif aov_name == "aov_normal_obj":
            out = tree.nodes.new("CompositorNodeOutputFile")
            out.base_path = os.path.join(tmp_root, "normal_obj")
            out.format.file_format = "PNG"
            out.format.color_mode = "RGBA"
            out.format.color_depth = "8"
            out.format.compression = 85
            out.file_slots[0].path = "frame_"
            if alpha_socket_name is not None:
                normal_set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
                links.new(rl.outputs[socket_name], normal_set_alpha.inputs["Image"])
                links.new(rl.outputs[alpha_socket_name], normal_set_alpha.inputs["Alpha"])
                links.new(normal_set_alpha.outputs["Image"], out.inputs[0])
            else:
                links.new(rl.outputs[socket_name], out.inputs[0])

    def _write_value_rgba(value_socket, folder_name):
        rgba = tree.nodes.new("CompositorNodeCombRGBA")
        links.new(value_socket, rgba.inputs["R"])
        links.new(value_socket, rgba.inputs["G"])
        links.new(value_socket, rgba.inputs["B"])
        if alpha_socket_name is not None:
            links.new(rl.outputs[alpha_socket_name], rgba.inputs["A"])
        else:
            rgba.inputs["A"].default_value = 1.0

        out = tree.nodes.new("CompositorNodeOutputFile")
        out.base_path = os.path.join(tmp_root, folder_name)
        out.format.file_format = "PNG"
        out.format.color_mode = "RGBA"
        out.format.color_depth = "8"
        out.format.compression = 100
        out.file_slots[0].path = "frame_"
        links.new(rgba.outputs["Image"], out.inputs[0])

    # ORM in RGBA: R=Occlusion, G=Roughness, B=Metallic, A=Render Alpha
    if {"aov_roughness", "aov_metallic", "aov_occlusion"}.issubset(socket_map.keys()):
        orm = tree.nodes.new("CompositorNodeCombRGBA")

        links.new(rl.outputs[socket_map["aov_occlusion"]], orm.inputs["R"])
        links.new(rl.outputs[socket_map["aov_roughness"]], orm.inputs["G"])
        links.new(rl.outputs[socket_map["aov_metallic"]], orm.inputs["B"])
        if alpha_socket_name is not None:
            links.new(rl.outputs[alpha_socket_name], orm.inputs["A"])
        else:
            orm.inputs["A"].default_value = 1.0

        orm_out = tree.nodes.new("CompositorNodeOutputFile")
        orm_out.base_path = os.path.join(tmp_root, "ORM")
        orm_out.format.file_format = "PNG"
        orm_out.format.color_mode = "RGBA"
        orm_out.format.color_depth = "8"
        orm_out.format.compression = 100
        orm_out.file_slots[0].path = "frame_"
        links.new(orm.outputs["Image"], orm_out.inputs[0])
    else:
        print("[WARN] Missing roughness/metallic/occlusion AOV, skip ORM output.")

    if label_socket_name is not None:
        label_scale = tree.nodes.new("CompositorNodeMath")
        label_scale.operation = "MULTIPLY"
        label_scale.inputs[1].default_value = 1.0 / 255.0
        links.new(rl.outputs[label_socket_name], label_scale.inputs[0])

        label_out = tree.nodes.new("CompositorNodeOutputFile")
        label_out.base_path = os.path.join(tmp_root, "label")
        label_out.format.file_format = "PNG"
        label_out.format.color_mode = "BW"
        label_out.format.color_depth = "8"
        label_out.format.compression = 100
        label_out.file_slots[0].path = "frame_"
        links.new(label_scale.outputs[0], label_out.inputs[0])
    else:
        raise RuntimeError(
            "Material index pass is enabled but no material-index socket was found. "
            f"Available RenderLayer sockets: {available}"
        )

    for d in ["images", "albedo", "depth", "ORM", "normal_mat", "normal_obj", "label"]:
        os.makedirs(os.path.join(tmp_root, d), exist_ok=True)


def finalize_output_layout(output_dir, frames_data):
    tmp_root = os.path.join(output_dir, "_tmp_channels")
    final_dirs = ["images", "albedo", "depth", "ORM", "normal_mat", "normal_obj", "label"]
    for d in final_dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    files = {
        "images": _list_sorted_files(os.path.join(tmp_root, "images"), ".png"),
        "albedo": _list_sorted_files(os.path.join(tmp_root, "albedo"), ".png"),
        "normal_mat": _list_sorted_files(os.path.join(tmp_root, "normal_mat"), ".png"),
        "normal_obj": _list_sorted_files(os.path.join(tmp_root, "normal_obj"), ".png"),
        "ORM": _list_sorted_files(os.path.join(tmp_root, "ORM"), ".png"),
        "label": _list_sorted_files(os.path.join(tmp_root, "label"), ".png"),
        "depth": _list_sorted_files(os.path.join(tmp_root, "depth"), ".exr"),
    }
    expected = len(frames_data)
    for k, arr in files.items():
        if len(arr) != expected:
            raise RuntimeError(f"{k} file count mismatch: expected {expected}, got {len(arr)}")

    for i in range(expected):
        stem = f"{i:03d}"
        shutil.move(files["images"][i], os.path.join(output_dir, "images", f"{stem}.png"))
        shutil.move(files["albedo"][i], os.path.join(output_dir, "albedo", f"{stem}.png"))
        shutil.move(files["normal_mat"][i], os.path.join(output_dir, "normal_mat", f"{stem}.png"))
        shutil.move(files["normal_obj"][i], os.path.join(output_dir, "normal_obj", f"{stem}.png"))
        shutil.move(files["ORM"][i], os.path.join(output_dir, "ORM", f"{stem}.png"))
        shutil.move(files["label"][i], os.path.join(output_dir, "label", f"{stem}.png"))
        shutil.move(files["depth"][i], os.path.join(output_dir, "depth", f"{stem}.exr"))

    shutil.rmtree(tmp_root, ignore_errors=True)
    return frames_data


def write_metadata_json(output_dir, case_obj_path, case_render_cfg, case_variant_map, label_id_map):
    _ = (case_obj_path, label_id_map)
    used = {}
    for key, material in sorted((case_variant_map or {}).items()):
        fam = _material_family_from_name(material)
        used[key] = {
            "material": material,
            "material_family": fam,
            "label_id": int(MATERIAL_SEGMENTATION_MAP.get(fam, 0)),
        }

    meta = {
        "variant_map": dict(case_variant_map or {}),
        "used_materials": used,
        "render_summary": {
            "resolution": case_render_cfg.get("resolution", {}),
            "samples": bpy.context.scene.cycles.samples if hasattr(bpy.context.scene, "cycles") else None,
            "output_structure": ["albedo", "images", "depth", "ORM", "normal_mat", "normal_obj", "label"],
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Single-pass AOV feasibility test")
    parser.add_argument("--config", help="Path to dataset_generation_config.yaml")
    parser.add_argument("--run_all", action="store_true", help="Run all enabled objects and all variants from config")
    parser.add_argument("--object_name", help="Object name in config")
    parser.add_argument("--variant_id", type=int, default=0, help="Variant index in generated variants list")
    parser.add_argument("--material_key", help="Optional: render only one material key as uniform material")
    parser.add_argument("--obj_path", help="Path to mesh file (.obj/.glb), direct mode")
    parser.add_argument("--material_dir", help="Material texture folder, direct mode")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--width", type=int, help="Render width")
    parser.add_argument("--height", type=int, help="Render height")
    args, _ = parser.parse_known_args(parse_blenderproc_args("generate_dataset.py"))

    obj_path = args.obj_path
    material_dir = args.material_dir
    output_dir = args.output_dir
    width = args.width if args.width is not None else 512
    height = args.height if args.height is not None else 512

    def render_one_case(
        case_obj_path,
        case_output_dir,
        case_render_cfg,
        case_material_dir=None,
        case_variant_map=None,
        case_manual_cfg=None,
        case_materials_root=None,
    ):
        bproc.clean_up()
        bproc.utility.reset_keyframes()

        objs = bproc.loader.load_obj(case_obj_path)
        if not objs:
            raise RuntimeError(f"Failed to load mesh: {case_obj_path}")

        label_id_map = build_label_id_map(case_variant_map or {})
        if case_variant_map is not None:
            assign_materials_manual(objs, case_manual_cfg or {}, case_variant_map, case_materials_root)
        else:
            mat = load_pbr_material("test_mat", case_material_dir)
            mat_name = os.path.basename(os.path.normpath(case_material_dir or ""))
            fam = _material_family_from_name(mat_name)
            fam_id = int(MATERIAL_SEGMENTATION_MAP.get(fam, 0))
            assign_uniform_material(objs, mat, label_id=fam_id)
            label_id_map = {"mat1": fam_id}
        frames_data = setup_camera_and_light(objs, case_render_cfg)

        aov_defs = add_shader_aov_outputs()
        rgb_cfg = case_render_cfg.get("rgb", {})
        _ = rgb_cfg  # kept for compatibility
        setup_compositor_outputs(
            case_output_dir,
            aov_defs,
            rgb_exposure=float(rgb_cfg.get("exposure", 1.0)),
            depth_scale=int(case_render_cfg.get("depth_scale", 5000)),
            depth_max_m=50.0,
        )

        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.cycles.samples = int(case_render_cfg.get("samples", 32))
        tmp_primary_dir = os.path.join(case_output_dir, "_tmp_primary_render")
        os.makedirs(tmp_primary_dir, exist_ok=True)
        bpy.context.scene.render.filepath = os.path.join(tmp_primary_dir, "frame_")
        bpy.ops.render.render(animation=True, write_still=False)
        shutil.rmtree(tmp_primary_dir, ignore_errors=True)
        frames_data = finalize_output_layout(case_output_dir, frames_data)
        dedup_cfg = case_render_cfg.get("image_matching", {})
        frames_data = deduplicate_rendered_views(case_output_dir, frames_data, dedup_cfg)
        write_transforms_json(case_output_dir, frames_data)
        write_material_segmentation_table(case_output_dir)
        write_metadata_json(case_output_dir, case_obj_path, case_render_cfg, case_variant_map, label_id_map)
        print(f"[OK] Single-pass render written to: {case_output_dir}")

    if args.config:
        cfg = load_yaml_config(args.config)
        global_cfg = cfg["global"]
        render_cfg = cfg.get("render", {})
        if args.width is not None:
            render_cfg.setdefault("resolution", {})
            render_cfg["resolution"]["width"] = int(args.width)
        if args.height is not None:
            render_cfg.setdefault("resolution", {})
            render_cfg["resolution"]["height"] = int(args.height)

        enabled_objects = [o for o in cfg.get("objects", []) if o.get("enabled", True)]
        if not enabled_objects:
            raise ValueError("No enabled objects found in config")

        if args.run_all:
            variant_cfg = global_cfg.get("variant_generation", {})
            strategy = variant_cfg.get("strategy", "cartesian")
            max_variants = variant_cfg.get("max_variants", 50)

            bproc.init()
            for obj_cfg in enabled_objects:
                variants = build_variant_maps(obj_cfg.get("material_candidates", {}), strategy, max_variants)
                for variant_id, variant_map in enumerate(variants):
                    case_obj_path = obj_cfg["obj_path"]
                    obj_out = obj_cfg.get("output_subdir", obj_cfg["name"])
                    case_output_dir = os.path.join(
                        global_cfg["output_root"],
                        obj_out,
                        f"var{variant_id:03d}",
                    )
                    print(
                        f"[RunAll] object={obj_cfg.get('name')} variant_id={variant_id} "
                        f"variant_map={variant_map}"
                    )
                    render_one_case(
                        case_obj_path,
                        case_output_dir,
                        render_cfg,
                        case_variant_map=variant_map,
                        case_manual_cfg=obj_cfg.get("manual", {}),
                        case_materials_root=global_cfg["materials_root"],
                    )
            return

        if args.object_name:
            obj_cfg = next((o for o in enabled_objects if o.get("name") == args.object_name), None)
            if obj_cfg is None:
                raise KeyError(f"object_name '{args.object_name}' not found in enabled objects")
        else:
            obj_cfg = enabled_objects[0]

        variant_cfg = global_cfg.get("variant_generation", {})
        strategy = variant_cfg.get("strategy", "cartesian")
        max_variants = variant_cfg.get("max_variants", 50)
        variants = build_variant_maps(obj_cfg.get("material_candidates", {}), strategy, max_variants)
        if not variants:
            raise ValueError("No variants generated from material_candidates")
        if args.variant_id < 0 or args.variant_id >= len(variants):
            raise IndexError(f"variant_id {args.variant_id} out of range [0, {len(variants)-1}]")

        variant_map = variants[args.variant_id]
        if not variant_map:
            raise ValueError("Variant map is empty, cannot select material for test")

        obj_path = obj_cfg["obj_path"]
        if args.material_key:
            if args.material_key not in variant_map:
                raise KeyError(f"material_key '{args.material_key}' not in variant map keys {list(variant_map.keys())}")
            material_name = variant_map[args.material_key]
            material_dir = os.path.join(global_cfg["materials_root"], material_name)
            if output_dir is None:
                obj_out = obj_cfg.get("output_subdir", obj_cfg["name"])
                output_dir = os.path.join(
                    global_cfg["output_root"],
                    obj_out,
                    f"var{args.variant_id:03d}_{args.material_key}-{material_name}",
                )
            print(
                f"[ConfigMode] object={obj_cfg.get('name')} variant_id={args.variant_id} "
                f"material_key={args.material_key} material={material_name}"
            )
        else:
            if output_dir is None:
                obj_out = obj_cfg.get("output_subdir", obj_cfg["name"])
                output_dir = os.path.join(
                    global_cfg["output_root"],
                    obj_out,
                    f"var{args.variant_id:03d}",
                )
            print(f"[ConfigMode] object={obj_cfg.get('name')} variant_id={args.variant_id} variant_map={variant_map}")
            obj_manual = obj_cfg.get("manual", {})
            obj_materials_root = global_cfg["materials_root"]

    if not obj_path or not output_dir:
        raise ValueError(
            "Missing inputs. Use --config mode, or provide --obj_path --material_dir --output_dir in direct mode."
        )

    bproc.init()
    if args.config and not args.material_key:
        render_one_case(
            obj_path,
            output_dir,
            render_cfg,
            case_variant_map=variant_map,
            case_manual_cfg=obj_manual,
            case_materials_root=obj_materials_root,
        )
    else:
        if not material_dir:
            raise ValueError("material_dir is required in direct mode or when using --material_key")
        direct_render_cfg = {
            "resolution": {"width": int(width), "height": int(height)},
            "camera": {"azimuth_deg": [0, 90, 180, 270], "elevation_deg": [0, 30]},
            "radius": {"min": 1.0, "max": None},
            "rgb": {"view_transform": "Standard", "exposure": 1.0},
            "lighting": {
                "mode": "fixed_three_point",
                "ambient_strength": 0.10,
                "ceiling_energy": 400.0,
                "ceiling_size": 2.5,
                "spot_energy": 150.0,
                "spot_height": 0.8,
                "spot_radius": 1.3,
                "spot_size_deg": 75.0,
                "spot_blend": 0.35,
                "spot_soft_size": 0.8,
            },
            "image_matching": {
                "enabled": True,
                "ssim_distance_threshold": 0.05,
                "clip_distance_threshold": 0.05,
            },
        }
        render_one_case(obj_path, output_dir, direct_render_cfg, case_material_dir=material_dir)


if __name__ == "__main__":
    main()
