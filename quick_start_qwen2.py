import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path

local_model_path = "./models/colqwen2-v1.0-merged"
pdf_paths = ["./test_data/2407.01449v6.pdf"]

model = ColQwen2.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(local_model_path)

# Your inputs
# images = [
#     Image.new("RGB", (128, 128), color="white"),
#     Image.new("RGB", (64, 32), color="black"),
# ]
images = []  # 最终存储所有PDF页的扁平列表（每个元素是单页PIL.Image）
for pdf_path in pdf_paths:
    # 转换PDF为图像列表（每页对应一个PIL.Image）
    # 可添加参数调整图像质量，例如 dpi=300 提高分辨率
    pdf_pages = convert_from_path(pdf_path, dpi=200)  # 返回：[page1_img, page2_img, ...]
    # 逐个添加每页图像到总列表（扁平化处理）
    for page_img in pdf_pages:
        images.append(page_img)

queries = [
    "What is ColPali?"
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

# 设置要获取的前k个匹配
k = 3  # 可根据需要调整
# 对每个查询，获取前k个最高分数的图像索引
top_k_values, top_k_indices = torch.topk(scores, k=k, dim=1)
# 打印每个查询的前k个匹配图像索引和分数
for i, (query_indices, query_scores) in enumerate(zip(top_k_indices, top_k_values)):
    print(f"\n查询 {i+1} 的前 {k} 个匹配图像:")
    for rank, (img_idx, score) in enumerate(zip(query_indices, query_scores), 1):
        print(f"  排名 {rank}: 图像索引={img_idx.item()}, 分数={score.item():.4f}")