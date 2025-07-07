from huggingface_hub import snapshot_download
import os

# 设置 Hugging Face 镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型名称
model_name = "vidore/colqwen2-v1.0"

# 指定下载目录
download_dir = "./models/local_colqwen2-v1.0"

# 下载模型和处理器文件
snapshot_download(repo_id=model_name, local_dir=download_dir, local_dir_use_symlinks=False)