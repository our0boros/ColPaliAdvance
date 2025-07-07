import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor, ColPali, ColPaliProcessor
import os

# 禁用所有网络请求，强制使用本地文件（保持不变）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # 新增，彻底禁用huggingface hub网络请求

# 本地模型路径（替换为colpali的本地目录，无需基础模型路径）
local_model_path = "./models/colpali-v1.3-merged"  # 确保该目录下有从vidore/colpali下载的所有文件

# 加载模型（替换为ColPali，移除base_model_path参数）
model = ColPali.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # 无GPU可改为"cpu"
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    local_files_only=True,  # 强制使用本地文件
    trust_remote_code=True  # 必需，colpali含自定义模型代码
).eval()

# 加载处理器（替换为ColPaliProcessor，无需base_model_path）
processor = ColPaliProcessor.from_pretrained(
    local_model_path,
    local_files_only=True  # 强制使用本地文件
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
print(scores)