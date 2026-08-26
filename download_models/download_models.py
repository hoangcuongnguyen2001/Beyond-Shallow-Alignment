from huggingface_hub import snapshot_download, login

login(token='')

snapshot_download(
    repo_id="meta-llama/Llama-Guard-3-8B",
    local_dir="/workspace/models/llama-guard",
    repo_type="model"
)