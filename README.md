# MatPredict 📦🪄  
*A synthetic benchmark for visual material-property learning in indoor robotics.*

This repository contains **(1)** scripts to re-generate the dataset from scratch, **(2)** training code for the baseline models published in our ICCV 2025 paper, and **(3)** environment files for reproducibility.

---

## 1  Up-stream Resources

| Resource | Link | What we use it for |
|----------|------|--------------------|
| **MatSynth** (4000 + PBR materials, CC-0) | <https://huggingface.co/datasets/gvecchio/MatSynth>| Randomly sampled to create diverse material stacks (base-colour, roughness, metallic, etc.). |
| **Replica** (18 photorealistic indoor scans) | <https://github.com/facebookresearch/Replica-Dataset> | Source of high-quality object meshes and HDR textures. |

---

## 2  MatPredict Dataset on Hugging Face

| Split | Size | Direct link |
|-------|------|-------------|
| **Full v1.0** | 120 GB | <https://huggingface.co/datasets/UMTRI/MatPredict> | 

Quick download (Python):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="UMTRI/MatPredict",
                filename="MatPredict_dataset.tar.zst",
                repo_type="dataset",
                local_dir="./data")
```


 
## 3 Environment — create from YAML

### create the conda environment
conda env create -f environment.yaml
conda activate matpredict

#### sanity check
```python
python -c "import torch, blenderproc; print('✅  environment ready')"
environment.yaml installs
```

Python 3.10

PyTorch ≥ 2.2 (CUDA 11.8)

BlenderProc 3.3

OpenCV, PyYAML, tqdm …



## 4 Generate the Dataset Locally
### activate env
conda activate matpredict

### clone this repo
git clone git@github.com:arpan-kusari/MatPredict.git
cd MatPredict

### tell BlenderProc where the source assets live
export REPLICA_ROOT=/path/to/Replica-Dataset
export MATSYNTH_ROOT=/path/to/MatSynth

### launch the generator
```python
python blender_render_code/generate_dataset.py
```

### What the script does

- item Extract each foreground mesh from Replica.

- item Smart-unwrap UVs & normalise texel density.

- item Attach a random MatSynth material.

- item Set up BlenderProc scene (lights + spherical camera rig).

- item Render RGB / normal / depth images in parallel.

- item Rendering ≈ 40 min per object on a 12-core CPU + RTX 4070 GPU.


## 5 Train a Baseline Model
### example: Restnet50 backbone
```python
python train.py
# remember to change the configuration parameters inside the python code
```
### the model will be saved at model/ directory 

## 6 Test a Baseline Model 
```python
python test_benchmarks.py
# remember to change the configuration parameters inside the python code
```

## 7 Citation
```bibtex
@dataset{chen2025matpredict,
  title  = {MatPredict: A Dataset and Benchmark for Learning Material Properties of Diverse Indoor Objects},
  author = {Yuzhen Chen and Hojun Son and Arpan Kusari},
  year   = {2025},
  url    = {https://huggingface.co/datasets/UMTRI/MatPredict}
}
```
 
