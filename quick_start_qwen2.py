import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor
import os

# 禁用所有网络请求，强制使用本地文件
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 告诉transformers库不联网
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 本地模型路径（主模型和基础模型）
local_model_path = "./models/colqwen2-v1.0"
local_base_model_path = "./models/colqwen2-base"  # 基础模型本地路径

# 加载模型（指定基础模型路径，避免自动联网下载）
model = ColQwen2.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    local_files_only=True,  # 强制使用本地文件
    base_model_path=local_base_model_path  # 手动指定基础模型本地路径（关键！）
).eval()

# 加载处理器（同样指定本地路径）
processor = ColQwen2Processor.from_pretrained(
    local_model_path,
    local_files_only=True,
    base_model_path=local_base_model_path  # 处理器也依赖基础模型配置
)

# 查看模型支持的方法
print(dir(model))  # 输出模型所有可用方法
# 检查processor是否提供高级功能
print(dir(processor))

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
