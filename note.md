![image-20250708142240800](note.assets/image-20250708142240800.png)

![image-20250708142257400](note.assets/image-20250708142257400.png)

满了...





切换为 Qwen2 的backbone

目前提出假想：

>   这种隐式的概念关联性是否真实存在？比如我询问：告诉我目前A值最高的模型是什么以及它能最高的原因。 这时候 RAG 提供的文档中应该包含 A值 模型评分图标信息 以及 A值的 概念性解释 ： A 是由 B 提出的 基于 C 构建的 D 框评分标准。 而 C 又会有 概念性解释： C 框架依托于 E 具体算法为 F。 这样一个 递归式的解释文档。以及一堆其他 相关领域的各种概念的 文档。 这时候现有 RAG 模型能否发现这一条明确的逻辑线路？ 参考 ColPali 由于存在 公共维度映射 图片信息，只要是 和 A 概念相关领域的 内容 都会有个 较高的 相关分数 它能明确出这一条逻辑线路吗？ 以及我如何评估验证是否存在这一逻辑链 ？

先简单的将ColPali这篇论文丢给模型让其判断哪些内容时最与 ColPali相关的，我们得到：

```
4070测试：
tensor([[15.6875, 16.0000, 11.1875,  8.5000, 15.2500, 14.0000, 14.6250, 14.6250,
         14.1875, 15.3750, 10.2500,  9.8125,  9.3125,  9.7500,  8.6875,  7.8438,
          8.0625, 14.9375, 13.6250,  5.9062, 14.8750, 11.4375,  6.4062,  6.9688,
          7.2500, 15.4375]])

查询 1 的前 3 个匹配图像:
  排名 1: 图像索引=1, 分数=16.0000
  排名 2: 图像索引=0, 分数=15.6875
  排名 3: 图像索引=25, 分数=15.4375
  
3090测试：
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
tensor([[15.6875, 16.1250, 11.1875,  8.5000, 15.2500, 14.0000, 14.5625, 14.6250,
         14.2500, 15.3750, 10.1875,  9.8125,  9.3750,  9.6875,  8.6250,  7.8125,
          8.1250, 15.0000, 13.8125,  5.9062, 14.8125, 11.4375,  6.3750,  6.9375,
          7.1562, 15.5625]])

查询 1 的前 3 个匹配图像:
  排名 1: 图像索引=1, 分数=16.1250
  排名 2: 图像索引=0, 分数=15.6875
  排名 3: 图像索引=25, 分数=15.5625
```

考虑到 ColPali 与 RAG 和 VLM 存在关联 索引度最高的 文件中也存在 相关描述，添加两篇相关论文查看是否存在最高值的索引

```
root@I228c2aae0800a01fde:/hy-tmp/ColPaliAdvance# vim qwen2_token_pooling.py 
root@I228c2aae0800a01fde:/hy-tmp/ColPaliAdvance# python qwen2_token_pooling.py 
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
You have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.
已加载缓存: 1 个PDF文件
从缓存加载: ./test_data/2407.01449v6.pdf
处理新PDF: ./test_data/2105.09996v3.pdf
已保存缓存到 ./cache/embeddings_cache.json
处理新PDF: ./test_data/2005.11401.pdf
已保存缓存到 ./cache/embeddings_cache.json

查询: What is ColPali?
排名 1: 相似度=0.2732, PDF: 2407.01449v6.pdf, 页面: 2
排名 2: 相似度=0.2432, PDF: 2407.01449v6.pdf, 页面: 22
排名 3: 相似度=0.2052, PDF: 2407.01449v6.pdf, 页面: 23

查询: How does ColPali work?
排名 1: 相似度=0.2618, PDF: 2407.01449v6.pdf, 页面: 2
排名 2: 相似度=0.2321, PDF: 2407.01449v6.pdf, 页面: 23
排名 3: 相似度=0.2299, PDF: 2407.01449v6.pdf, 页面: 22
root@I228c2aae0800a01fde:/hy-tmp/ColPaliAdvance# 

```

目前看来是不存在的，因此考虑是否通过这一部分的优化能实现更高精度的索引

