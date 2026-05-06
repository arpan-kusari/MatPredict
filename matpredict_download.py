from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="anonymous-submission/MatPredict",
    repo_type="dataset",
    local_dir="/path/to/MatPredictDataset",
    local_dir_use_symlinks=False,
    max_workers=4,
)
