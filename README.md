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
| **Full v1.0** | 110 GB | <https://huggingface.co/datasets/UMTRI/MatPredict> |
| **Mini (v1.0 1-scene subset)** | 3.2 GB | <https://huggingface.co/datasets/UMTRI/MatPredict/tree/main/mini> |

Quick download (Python):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="UMTRI/MatPredict",
                filename="mini.tar.zst",
                repo_type="dataset",
                local_dir="./data")
