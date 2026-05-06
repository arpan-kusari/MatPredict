# MatPredict

MatPredict is a synthetic benchmark for learning material properties of indoor
objects from RGB images. The current codebase contains tools for dataset
generation and benchmark models for two tasks:

- inverse rendering: predict albedo, roughness, and metallic maps
- material segmentation: predict per-pixel material classes

## Resources

| Resource | Link | Use |
| --- | --- | --- |
| MatPredict dataset | <https://huggingface.co/datasets/UMTRI/MatPredict> | Pre-generated dataset and benchmark data |
| MatSynth | <https://huggingface.co/datasets/gvecchio/MatSynth> | Source PBR material library |
| ReplicaCAD assets | <https://aihabitat.org/datasets/replica_cad> | Source object meshes |

## Repository Layout

```text
MatPredict/
  matpredict_download.py              # Download the hosted MatPredict dataset
  data_generation/
    configs/dataset_generation_config.yaml
    generate_dataset.py               # BlenderProc dataset generator
  benchmarks/
    configs/                          # Training/evaluation/inference templates
    dataset/                          # Split generation and dataset loader
    resnet50/                         # ResNet50 U-Net benchmark model
    swin_t/                           # Swin-T U-Net benchmark model
    scripts/                          # Train, evaluate, and single-image inference
```

## Download The Dataset

Install the Hugging Face helper and edit `local_dir` in
`matpredict_download.py` if needed:

```bash
pip install huggingface_hub
python matpredict_download.py
```

By default the script downloads to:

```text
/path/to/MatPredictDataset
```

## Dataset Structure

The downloaded dataset is organized as:

```text
MatPredictDataset/
  material_segmentation_map.yaml
  material_segmentation_train_config.yaml
  config/
    object_disjoint_v1.yaml
    variance_disjoint_v1.yaml
  <object_name>/
    <variant_name>/
      images/*.png
      albedo/*.png
      ORM/*.png
      depth/*.exr
      normal_mat/*.png
      normal_obj/*.png
      label/*.png
      transforms.json
      metadata.json
      material_segmentation_map.json
```

Each frame uses the same zero-padded id across folders, for example
`images/000.png`, `albedo/000.png`, `ORM/000.png`, and `label/000.png`.

The benchmark loader uses `images`, `albedo`, `ORM`, and `label`. `ORM` stores
occlusion/roughness/metallic channels, with roughness in the green channel and
metallic in the blue channel. `label` is used as the foreground mask for inverse
rendering and as the material class map for material segmentation.

The additional `depth`, `normal_mat`, `normal_obj`, `transforms.json`, and
`metadata.json` files are included for geometry-aware analysis, camera metadata,
and reproducibility.

## Generate Data

Dataset generation uses BlenderProc and a manual material-slot mapping config.
Create the generation environment first:

```bash
conda create -n blenderproc python=3.12 -y
conda activate blenderproc
pip install -r data_generation/requirements.txt
```

Edit:

```text
data_generation/configs/dataset_generation_config.yaml
```

The example config contains:

- one single-material object example
- one composited / multi-slot object example

Run one configured object/variant:

```bash
blenderproc run data_generation/generate_dataset.py -- \
  --config data_generation/configs/dataset_generation_config.yaml \
  --object_name single_material_example \
  --variant_id 0
```

Run all enabled example objects and generated variants:

```bash
blenderproc run data_generation/generate_dataset.py -- \
  --config data_generation/configs/dataset_generation_config.yaml \
  --run_all
```

Generated outputs include RGB images, albedo, depth, ORM, normals, labels,
`transforms.json`, and metadata.

## Benchmark Environment

```bash
cd benchmarks
conda create -n matpredict_bench python=3.10 -y
conda activate matpredict_bench
pip install -r requirements.txt
```

## Generate Splits

```bash
cd benchmarks
python dataset/generate_splits.py \
  --dataset_root /path/to/MatPredictDataset \
  --output_dir ./dataset/splits \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

This creates two benchmark protocols:

- `object_disjoint_v1.yaml`: object ids are disjoint across train, val, and test
- `variance_disjoint_v1.yaml`: variant ids are disjoint across train, val, and test

## Train

Use the template config:

```bash
cd benchmarks
python scripts/train.py --config ./configs/training_template.yaml
```

Important fields in `configs/training_template.yaml`:

- `dataset_root`: path to `MatPredictDataset`
- `split_file`: choose object-disjoint or variance-disjoint split
- `model_name`: `resnet50_unet` or `swin_t_unet`
- `task_mode`: `pbr` for inverse rendering or `material` for material
  segmentation
- `material_num_classes`: set this for material segmentation, for example `8`

CLI arguments override config values:

```bash
python scripts/train.py \
  --config ./configs/training_template.yaml \
  --model_name swin_t_unet \
  --task_mode material \
  --material_num_classes 8
```

## Evaluate

Edit `configs/evaluation_template.yaml` so `model_name`, `task_mode`, and
`checkpoint` match the trained checkpoint:

```bash
cd benchmarks
python scripts/evaluation.py --config ./configs/evaluation_template.yaml
```

Evaluation writes:

```text
<output_dir>/metrics_<split>.yaml
```

## Single-Image Inference

Edit `configs/inference_template.yaml`:

```bash
cd benchmarks
python scripts/inference.py --config ./configs/inference_template.yaml
```

For `task_mode: pbr`, inference writes:

- `pred_albedo.png`
- `pred_roughness.png`
- `pred_metallic.png`
- `panel.png`

For `task_mode: material`, inference writes:

- `pred_material_ids.png`
- `pred_material_color.png`
- `panel.png`

<!-- ## Citation

```bibtex
@dataset{chen2025matpredict,
  title  = {MatPredict: A Dataset and Benchmark for Learning Material Properties of Diverse Indoor Objects},
  author = {Yuzhen Chen and Hojun Son and Arpan Kusari},
  year   = {2025},
  url    = {https://huggingface.co/datasets/UMTRI/MatPredict}
}
``` -->
