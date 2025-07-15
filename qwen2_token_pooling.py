import torch
import os
import json
import time
import gc
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from typing import List, Dict, Tuple, Optional


class ColPaliRetriever:
    def __init__(self, model_path: str, cache_dir: str = "./cache"):
        """初始化ColPali检索器并管理缓存"""
        self.model = ColQwen2.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 使用float16降低内存占用
            device_map="cuda:0",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            gradient_checkpointing=True,  # 启用梯度检查点减少内存
        ).eval()

        self.processor = ColQwen2Processor.from_pretrained(model_path)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 缓存结构
        self.pdf_cache = {}  # 记录已处理的PDF及其页面嵌入
        self.cache_path = os.path.join(cache_dir, "embeddings_cache.json")

        # 加载现有缓存
        if os.path.exists(self.cache_path):
            self._load_cache()

    def _load_cache(self):
        """从磁盘加载缓存"""
        try:
            with open(self.cache_path, 'r') as f:
                self.pdf_cache = json.load(f)
            print(f"已加载缓存: {len(self.pdf_cache)} 个PDF文件")
        except Exception as e:
            print(f"加载缓存失败: {e}")
            self.pdf_cache = {}

    def _save_cache(self):
        """保存缓存到磁盘"""
        with open(self.cache_path, 'w') as f:
            json.dump(self.pdf_cache, f)
        print(f"已保存缓存到 {self.cache_path}")

    def process_pdfs(self, pdf_paths: List[str], dpi: int = 100) -> Dict[str, List[torch.Tensor]]:
        """
        处理PDF文件并计算/加载嵌入

        Args:
            pdf_paths: PDF文件路径列表
            dpi: PDF转图像的分辨率

        Returns:
            包含所有PDF页面嵌入的字典
        """
        all_embeddings = []
        page_to_pdf_map = []  # 记录每个页面属于哪个PDF

        for pdf_path in pdf_paths:
            pdf_hash = self._get_file_hash(pdf_path)

            # 检查缓存
            if pdf_hash in self.pdf_cache:
                print(f"从缓存加载: {pdf_path}")
                # 直接加载到GPU
                pdf_embeddings = [torch.tensor(e).to(self.model.device) for e in self.pdf_cache[pdf_hash]['embeddings']]
            else:
                print(f"处理新PDF: {pdf_path}")
                # 转换PDF为图像
                images = []
                try:
                    pdf_pages = convert_from_path(pdf_path, dpi=dpi)
                    for page_img in pdf_pages:
                        images.append(page_img)
                except Exception as e:
                    print(f"处理PDF失败: {pdf_path}, 错误: {e}")
                    continue

                # 计算嵌入
                if images:
                    batch_size = 5  # 每批次处理的页数
                    pdf_embeddings = []
                    for i in range(0, len(images), batch_size):
                        batch_images = images[i:i + batch_size]
                        processed_batch = self.processor.process_images(batch_images).to(self.model.device)
                        with torch.no_grad():
                            # 模型直接返回嵌入张量
                            outputs = self.model(**processed_batch)
                            batch_embeddings = self._pool_embeddings(outputs)
                            pdf_embeddings.extend(batch_embeddings)

                        # 清理内存
                        del processed_batch, outputs, batch_embeddings
                        torch.cuda.empty_cache()
                        gc.collect()

                    # 保存到缓存（转为CPU列表）
                    self.pdf_cache[pdf_hash] = {
                        'path': pdf_path,
                        'timestamp': time.time(),
                        'embeddings': [e.cpu().tolist() for e in pdf_embeddings]
                    }
                    self._save_cache()

            # 记录所有嵌入和映射关系
            all_embeddings.extend(pdf_embeddings)
            page_to_pdf_map.extend([(pdf_path, i) for i in range(len(pdf_embeddings))])

        return {
            'embeddings': all_embeddings,
            'page_to_pdf_map': page_to_pdf_map
        }

    def _pool_embeddings(self, hidden_states: torch.Tensor, strategy: str = "mean") -> List[torch.Tensor]:
        """
        应用token pooling策略获取固定长度的表示

        Args:
            hidden_states: 模型输出的嵌入张量 [batch_size, num_patches, embedding_dim]
            strategy: 池化策略 ("mean", "max")

        Returns:
            池化后的嵌入列表
        """
        if len(hidden_states.shape) != 3:
            raise ValueError(
                f"期望嵌入张量形状为 [batch_size, num_patches, embedding_dim]，但得到 {hidden_states.shape}")

        if strategy == "mean":
            # 平均池化：对所有patch的嵌入取平均
            return [hidden_states[i].mean(dim=0) for i in range(hidden_states.shape[0])]
        elif strategy == "max":
            # 最大池化：对所有patch的嵌入取最大值
            return [hidden_states[i].max(dim=0)[0] for i in range(hidden_states.shape[0])]
        else:
            raise ValueError(f"不支持的池化策略: {strategy}")

    def _get_file_hash(self, file_path: str) -> str:
        """生成文件的简单哈希（基于文件名和修改时间）"""
        try:
            mtime = os.path.getmtime(file_path)
            return f"{os.path.basename(file_path)}_{mtime}"
        except:
            # 如果无法获取文件信息，使用时间戳
            return f"{os.path.basename(file_path)}_{time.time()}"

    def retrieve(self, query: str, pdf_data: Dict, k: int = 3) -> List[Tuple[float, int, str]]:
        """
        检索与查询最匹配的页面

        Args:
            query: 查询文本
            pdf_data: 包含PDF嵌入和映射关系的数据
            k: 返回的结果数量

        Returns:
            包含(相似度分数, 页面索引, PDF路径)的列表
        """
        # 计算查询嵌入
        batch_queries = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            query_output = self.model(**batch_queries)
            query_embedding = self._pool_embeddings(query_output)[0]

        # 计算相似度
        similarities = []
        for i, page_embedding in enumerate(pdf_data['embeddings']):
            # 确保所有嵌入在同一设备上
            page_embedding = page_embedding.to(query_embedding.device)
            # 计算余弦相似度
            sim = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                page_embedding.unsqueeze(0)
            ).item()
            similarities.append((sim, i))

        # 排序并获取前k个结果
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]

        # 转换为最终结果格式
        results = []
        for score, page_idx in top_k:
            pdf_path, page_num = pdf_data['page_to_pdf_map'][page_idx]
            results.append((score, page_num, pdf_path))

        return results


# 使用示例
if __name__ == "__main__":
    local_model_path = "./models/colqwen2-v1.0-merged"
    pdf_paths = ["./test_data/2407.01449v6.pdf"]

    # 初始化检索器
    retriever = ColPaliRetriever(local_model_path)

    # 处理PDF文件（自动加载缓存或计算新嵌入）
    pdf_data = retriever.process_pdfs(pdf_paths)

    # 查询示例
    queries = [
        "What is ColPali?",
        "How does ColPali work?"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        # 检索最相关的页面
        results = retriever.retrieve(query, pdf_data, k=3)

        # 显示结果
        for rank, (score, page_num, pdf_path) in enumerate(results, 1):
            print(f"排名 {rank}: 相似度={score:.4f}, PDF: {os.path.basename(pdf_path)}, 页面: {page_num + 1}")