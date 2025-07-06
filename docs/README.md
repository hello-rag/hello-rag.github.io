<div align='center'>
    <img src="./images/hello-rag.png" alt="alt text" width="100%">
    <h1>Hello-RAG</h1>
</div>
<div align="center">
  <h3>🤖 大道至简：从零开始的RAG构建及应用教程</h3>
  <p><em>由浅入深，一步步带你搭建自己的RAG系统</em></p>
</div>
<!-- > ✨ 此仓库为中文的实现过程，并将每一步中文注释，且都跑通实现，目前基于文本进行实现，没有实现多模态的RAG。详细的内容可以去看 [RESOURCES](#RESOURCES) 哦。 -->

---

## 🏳️‍🌈 前言
`RAG（Retrieval-Augmented Generation）`：`RAG`是一种基于检索的生成模型，旨在通过检索到相关文本块来生成响应。具体来说，当模型需要生成文本或者回答问题时，它会先从一个庞大的文档集合中检索出相关的信息，然后利用这些检索到的信息来指导文本的生成，从而提高预测的质量和准确性。

`Hello-RAG` 采用从头搭建的方式，完全不依赖 `LangChain`、`LlamaIndex` 等现有框架。我们仅使用基础的 Python 库（如 `openai`、`numpy`、`fitz`（`pymupdf`）等），让开发者能够深入到 RAG 的每一个细节中，亲手构建每一个模块，从而对 RAG 的工作原理有更深入的理解。通过 `Hello-RAG`，开发者可以熟悉文本嵌入、语义检索、上下文处理和响应生成等各个流程，真正掌握 RAG 的核心。
## 🎈 内容导航
| 章节 | 关键内容 | 状态 |
| --- | --- | --- |
| [第一章 简单 RAG](./content/01_simple_rag.md) | 基本的 RAG 实现 |✅|
| [第二章 语义块切分](./content/02_semantic_chunking.md) | 根据语义相似性分割文本，以形成更有意义的块。 |✅|
| [第三章 上下文增强检索](./content/03_context_enriched_rag.md) | 获取相邻块以提供更多上下文。 |✅|
| [第四章 上下文分块标题](./content/04_contextual_chunk_headers.md) | 在嵌入之前，为每个片段添加描述性标题。 |✅|
| [第五章 文档增强 RAG](./content/05_doc_augmentation_rag.md) | 从文本片段生成问题以增强检索过程。 |✅|
| [第六章 查询转换](./content/06_query_transform.md) | 重新编写、扩展或分解查询以提高检索效果。包括回退提示和子查询分解。 |✅|
| [第七章 重新排序器](./content/07_reranker.md) | 使用 LLM 对最初检索到的结果进行重排，以获得更好的相关性。 |✅|
| [第八章 相关段落提取](./content/08_rse.md) | 识别并重建连续的文本段落，保留上下文。 |✅|
| [第九章 上下文压缩](./content/09_contextual_compression.md) | 实现上下文压缩以过滤和压缩检索到的块，最大化相关信息。 |✅|
| [第十章 反馈循环](./content/10_feedback_loop_rag.md) | 随时间推移，通过用户反馈学习并改进 RAG 系统。 |✅|
| [第十一章 适应性RAG](./content/11_adaptive_rag.md) | 根据查询类型动态选择最佳检索策略。 |✅|
| [第十二章 Self-RAG](./content/12_self_rag.md) | 动态决定何时以及如何检索，评估相关性，并评估支持和效用。 |✅|
| [第十三章 融合RAG](./content/13_fusion_rag.md) | 结合向量搜索和基于关键词（BM25）的检索，以改善结果。 |✅|
| [第十四章 图谱RAG](./content/14_graph_rag.md) | 将知识组织为图，使相关概念能够遍历。 |✅|
| [第十五章 层次索引](./content/15_hierarchy_rag.md) | 构建层次索引（摘要+详细片段），以实现高效检索。 |✅|
| [第十六章 HyDE RAG](./content/16_HyDE_rag.md) | 使用假设文档嵌入来提高语义匹配。 |✅|
| [第十七章 CRAG](./content/17_crag.md) | 动态评估检索质量，并使用网络搜索作为后备 |✅|
| **分块技术** | | |
| [块大小选择器](./content/001_chunk_size_selector.md)             | 探讨不同块大小对检索性能的影响。                             |✅|
| [命题分块](./content/00_chunk_size_selector.md)               | 将文档分解为原子事实陈述，以实现精确检索。                   |✅|
## 📚 关键概念
- **RAG（Retrieval-Augmented Generation）**：RAG是一种基于检索的生成模型，旨在通过检索到相关文本块来生成响应。RAG系统由三个主要组件组成：检索模块、块嵌入模块和生成模块。
- **块（Chunk）**：一种文本片段，通常由一组连续的句子组成。
- **嵌入（Embedding）**：一种向量表示，用于表示文本或文本片段。
- **上下文（Context）**：一种文本片段，通常由一组连续的句子组成，用于提供更多信息
- **向量存储(Vector Store)**: 存储和搜索嵌入的简单数据库。我们使用NumPy创建自己的SimpleVectorStore类进行高效相似度计算。
- **余弦相似度(Cosine Similarity)**: 衡量两个向量相似度的指标，值越高表示相似度越大。
- **分块(Chunking)**: 将文本分割为更小、更易管理的部分。我们探索多种分块策略。
- **检索(Retrieval)**: 为给定查询寻找最相关文本块的过程。
- **生成(Generation)**: 使用大语言模型(LLM)基于检索到的上下文和用户查询生成响应。
- **评估(Evaluation)**: 通过比较参考答案或使用LLM评分来评估RAG系统响应质量。


## 💡 学习建议

本项目适合大学生、研究人员、`LLM` 爱好者。在学习本项目之前，你需要具备一定的编程经验，尤其是要对 `Python` 编程语言有一定的了解。最好具备深度学习的相关知识，并了解 `RAG` 和 `LLM` 领域的相关概念和术语，以便更轻松地学习本项目。

本项目分为两部分——`RAG`和分块技术。第1章～第17章是`RAG`及其各种实现方法，从浅入深介绍 `RAG` 的基本原理。分块技术部分介绍了不同块大小对检索性能的影响，以及如何将文档分解为原子事实陈述，以实现精确检索。代码不是最主要的，重要的是理解 `RAG` 的原理，了解思路，并尝试自己实现一些模块。你可以根据个人兴趣和需求，选择性地阅读相关章节。

## 🔋 参考文档
- [all-rag-techniques](https://github.com/hemmydev/all-rag-techniques)
- [rag-all-techniques](https://github.com/liu673/rag-all-techniques)
- [Happy-LLM](https://github.com/datawhalechina/happy-llm)