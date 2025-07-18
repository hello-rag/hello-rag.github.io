# 增强RAG的关联段落提取（RSE）
![](../images/RSE.webp)
实现关联段落提取（Relevant Segment Extraction，RSE）技术，以提高RAG系统的上下文质量。我们不仅仅检索一组孤立的片段，而是识别并重建提供更好上下文的连续文本段落，从而为语言模型提供更好的支持。

-----
核心概念：

相关的片段往往会在文档中聚集成簇。通过识别这些簇并保持其连续性，RSE为大型语言模型提供了更加连贯的上下文。

-----
实现步骤：
- 处理文档以创建向量存储：从PDF 中提取文本，分割文本块（0重叠）并创建向量存储
- 根据查询计算相关性分数和块值：
    - 先获取所有带有相似度分数的块；
    - 获取相关性分数，如果不在结果中则默认为0，同时应用惩罚以将不相关的块转换为负值
- 根据块值找到最佳文本块：使用最大子数组和算法的变体找到最佳
- 从最佳块中重建文本段落：基于块索引重建文本段落
- 利用将段落格式化为上下文生成回答


```python
import fitz
import os
import re
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
```




    True




```python
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
llm_model = os.getenv("LLM_MODEL_ID")
embedding_model = os.getenv("EMBEDDING_MODEL_ID")

pdf_path = "../../data/AI_Information.en.zh-CN.pdf"
```


```python
def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取文本，并打印前 `num_chars` 个字符。

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # 打开 PDF 文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串以存储提取的文本

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")  # 从页面中提取文本
        all_text += text  # 将提取的文本追加到 all_text 字符串中

    return all_text  # 返回提取的文本
```


```python
def chunk_text(text, n, overlap):
    """
    将文本分割为重叠的块

    Args:
        text (str): 要分割的文本
        n (int): 每个块的字符数
        overlap (int): 块之间的重叠字符数

    Returns:
        List[str]: 文本块列表
    """
    chunks = []  #
    for i in range(0, len(text), n - overlap):
        # 添加从当前索引到索引 + 块大小的文本块
        chunk = text[i:i + n]
        if chunk:
            chunks.append(chunk)

    return chunks  # Return the list of text chunks
```


```python
class SimpleVectorStore:
    """
    使用 NumPy 实现的轻量级向量存储。
    """
    def __init__(self, dimension=1536):
        """
        初始化向量存储。

        Args:
            dimension (int): 嵌入向量的维度
        """
        self.dimension = dimension  # 嵌入向量的维度
        self.vectors = []  # 存储嵌入向量的列表
        self.documents = []  # 存储文档片段的列表
        self.metadata = []  # 存储元数据的列表

    def add_documents(self, documents, vectors=None, metadata=None):
        """
        向向量存储中添加文档。

        Args:
            documents (List[str]): 文档片段列表
            vectors (List[List[float]], 可选): 嵌入向量列表
            metadata (List[Dict], 可选): 元数据字典列表
        """
        if vectors is None:  # 如果未提供向量，则生成一个空列表
            vectors = [None] * len(documents)

        if metadata is None:  # 如果未提供元数据，则生成一个空字典列表
            metadata = [{} for _ in range(len(documents))]

        for doc, vec, meta in zip(documents, vectors, metadata):  # 遍历文档、向量和元数据
            self.documents.append(doc)  # 将文档片段添加到列表中
            self.vectors.append(vec)  # 将嵌入向量添加到列表中
            self.metadata.append(meta)  # 将元数据添加到列表中

    def search(self, query_vector, top_k=5):
        """
        搜索最相似的文档。

        Args:
            query_vector (List[float]): 查询嵌入向量
            top_k (int): 返回的结果数量

        Returns:
            List[Dict]: 包含文档、分数和元数据的结果列表
        """
        if not self.vectors or not self.documents:  # 如果向量或文档为空，返回空列表
            return []

        # 将查询向量转换为 NumPy 数组
        query_array = np.array(query_vector)

        # 计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):  # 遍历存储中的向量
            if vector is not None:  # 如果向量不为空
                # 计算余弦相似度
                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))  # 将索引和相似度添加到列表中

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取前 top-k 结果
        results = []
        for i, score in similarities[:top_k]:  # 遍历前 top-k 的结果
            results.append({
                "document": self.documents[i],  # 文档片段
                "score": float(score),  # 相似度分数
                "metadata": self.metadata[i]  # 元数据
            })

        return results  # 返回结果列表

```


```python
def create_embeddings(texts):
    """
    为文本生成嵌入向量。

    Args:
        texts (List[str]): 要嵌入的文本列表

    Returns:
        List[List[float]]: 嵌入向量列表
    """
    if not texts:  # 如果没有提供文本，返回空列表
        return []

    # 如果列表很长，则按批次处理
    batch_size = 100  # 根据API限制进行调整
    all_embeddings = []  # 初始化一个列表来存储所有嵌入向量

    for i in range(0, len(texts), batch_size):  # 按批次处理文本
        batch = texts[i:i + batch_size]  # 获取当前批次的文本

        # 使用指定的模型为当前批次生成嵌入向量
        response = client.embeddings.create(
            input=batch,  # 输入文本批次
            model=embedding_model  # 使用的模型
        )

        # 从响应中提取嵌入向量
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将批次嵌入向量添加到总列表中

    return all_embeddings  # 返回所有嵌入向量的列表

```

# 8.1 用 RSE 处理文档


```python
def process_document(pdf_path, chunk_size=800):
    """
    处理文档以供 RSE 使用。

    Args:
        pdf_path (str): PDF 文档的路径
        chunk_size (int): 每个块的字符大小

    Returns:
        Tuple[List[str], SimpleVectorStore, Dict]: 块列表、向量存储和文档信息
    """
    print("从文档中提取文本...")
    # 从 PDF 文件中提取文本
    text = extract_text_from_pdf(pdf_path)

    print("将文本切分为非重叠段落...")
    # 将提取的文本切分为非重叠段落
    chunks = chunk_text(text, n=chunk_size, overlap=0)
    print(f"创建了 {len(chunks)} 个块")

    print("为块生成嵌入向量...")
    # 为文本块生成嵌入向量
    chunk_embeddings = create_embeddings(chunks)

    # 创建 SimpleVectorStore 的实例
    vector_store = SimpleVectorStore()

    # 添加带有元数据的文档（包括块索引以便后续重建）
    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]
    vector_store.add_documents(chunks, chunk_embeddings, metadata)

    # 跟踪原始文档结构以便段落重建
    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }

    return chunks, vector_store, doc_info

```

# 8.2 RSE核心算法：计算块值和寻找最佳段落
现在我们已经有了处理文档和为其块生成嵌入向量所需的函数，可以实现RSE的核心算法。


```python
def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    通过结合相关性和位置计算块的值。

    Args:
        query (str): 查询文本
        chunks (List[str]): 文档块列表
        vector_store (SimpleVectorStore): 包含块的向量存储
        irrelevant_chunk_penalty (float): 不相关块的惩罚值

    Returns:
        List[float]: 块值列表
    """
    # 创建查询嵌入
    query_embedding = create_embeddings([query])[0]

    # 获取所有带有相似度分数的块
    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)

    # 创建从块索引到相关性分数的映射
    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}

    # 计算块值（相关性分数减去惩罚）
    chunk_values = []
    for i in range(num_chunks):
        # 获取相关性分数，如果不在结果中则默认为0
        score = relevance_scores.get(i, 0.0)
        # 应用惩罚以将不相关的块转换为负值
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)

    return chunk_values

```


```python
def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    使用最大子数组和算法的变体找到最佳段落。

    Args:
        chunk_values (List[float]): 每个块的值
        max_segment_length (int): 单个段落的最大长度
        total_max_length (int): 所有段落的最大总长度
        min_segment_value (float): 被考虑的段落的最小值

    Returns:
        List[Tuple[int, int]]: 最佳段落的（开始，结束）索引列表
    """
    print("寻找最佳连续文本段落...")

    best_segments = []
    segment_scores = []
    total_included_chunks = 0

    # 继续寻找段落直到达到限制
    while total_included_chunks < total_max_length:
        best_score = min_segment_value  # 段落的最低阈值
        best_segment = None

        # 尝试每个可能的起始位置
        for start in range(len(chunk_values)):
            # 如果该起始位置已经在选定的段落中，则跳过(重叠内容部分)
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue

            # 尝试每个可能的段落长度
            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length

                # 如果结束位置已经在选定的段落中，则跳过
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue

                # 计算段落值为块值的总和
                segment_value = sum(chunk_values[start:end])

                # 如果这个段落更好，则更新最佳段落
                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)

        # 如果找到了一个好的段落，则添加它
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            print(f"找到段落 {best_segment}，得分 {best_score:.4f}")
        else:
            # 没有更多的好段落可找
            break

    # 按段落的起始位置排序以便于阅读
    best_segments = sorted(best_segments, key=lambda x: x[0])

    return best_segments, segment_scores

```

# 8.3 为RAG重建和使用段落

通过重建和利用文本段落来改进上下文质量和生成效果。


```python
def reconstruct_segments(chunks, best_segments):
    """
    基于块索引重建文本段落。

    Args:
        chunks (List[str]): 所有文档块的列表
        best_segments (List[Tuple[int, int]]): 段落的（开始，结束）索引列表

    Returns:
        List[Dict]: 重建的文本段落列表，每个段落包含文本和其范围
    """
    reconstructed_segments = []  # 初始化一个空列表以存储重建的段落

    for start, end in best_segments:
        # 将此段落中的块连接起来以形成完整的段落文本
        segment_text = " ".join(chunks[start:end])
        # 将段落文本及其范围追加到重建的段落列表中
        reconstructed_segments.append({
            "text": segment_text,  # 段落文本
            "segment_range": (start, end),  # 段落范围
        })

    return reconstructed_segments  # 返回重建的文本段落列表

```


```python
def format_segments_for_context(segments):
    """
    将段落格式化为适用于LLM的上下文字符串。

    Args:
        segments (List[Dict]): 段落字典列表

    Returns:
        str: 格式化后的上下文文本
    """
    context = []  # 初始化一个空列表以存储格式化后的上下文

    for i, segment in enumerate(segments):
        # 为每个段落创建一个包含索引和块范围的标题
        # segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        segment_header = f"分段{i+1}（包含文本块{segment['segment_range'][0]}至{segment['segment_range'][1]-1}）："
        context.append(segment_header)  # 将段落标题添加到上下文列表中
        context.append(segment['text'])  # 将段落文本添加到上下文列表中
        context.append("-" * 80)  # 添加分隔线以提高可读性

    # 将上下文列表中的所有元素用双换行符连接并返回结果
    return "\n\n".join(context)

```

# 8.4 使用 RSE 上下文生成回答


```python
def generate_response(query, context):
    """
    根据查询和上下文生成响应。

    Args:
        query (str): 用户查询
        context (str): 来自相关段落的上下文文本

    Returns:
        str: 生成的响应
    """
    print("正在使用相关段落作为上下文生成响应...")

    # 定义系统提示以引导AI的行为
    system_prompt = """
    您是基于上下文智能应答的AI助手，需根据提供的文档段落回答用户问题。
    这些文档段落是通过相关性检索匹配到当前问题的上下文内容。
    请严格依据以下要求执行：
    1. 整合分析所有相关段落信息
    2. 生成全面准确的综合回答
    3. 当上下文不包含有效信息时，必须明确告知无法回答
    """

    # 通过组合上下文和查询创建用户提示
    user_prompt = f"""
    上下文内容：
    {context}

    问题：{query}

    请基于上述上下文内容提供专业可靠的回答。
    """

    # 使用指定的模型生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 返回生成的响应内容
    return response.choices[0].message.content

```

# 8.5 完整的RSE管道功能


```python
def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    """
    完整的RAG管道，包含相关段落提取（Relevant Segment Extraction）。

    Args:
        pdf_path (str): 文档路径
        query (str): 用户查询
        chunk_size (int): 每个块的大小
        irrelevant_chunk_penalty (float): 对不相关块的惩罚值

    Returns:
        Dict: 包含查询、段落和响应的结果字典
    """
    print("\n=== 开始带有相关段落提取的RAG ===")
    print(f"查询: {query}")

    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)

    # 根据查询计算相关性分数和块值
    print("\n计算相关性分数和块值...")
    chunk_values = calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty)

    # 根据块值找到最佳文本段落
    best_segments, scores = find_best_segments(
        chunk_values,
        max_segment_length=20,  # 最大段落长度
        total_max_length=30,  # 所有段落的最大总长度
        min_segment_value=0.2  # 考虑段落的最小值
    )

    # 从最佳块中重建文本段落
    print("\n从块中重建文本段落...")
    segments = reconstruct_segments(chunks, best_segments)

    # 将段落格式化为语言模型的上下文字符串
    context = format_segments_for_context(segments)

    # 使用上下文从语言模型生成响应
    response = generate_response(query, context)

    # 将结果编译成字典
    result = {
        "query": query,  # 用户查询
        "segments": segments,  # 提取的段落
        "response": response  # 模型生成的响应
    }

    print("\n=== 最终响应 ===")
    print(response)

    return result

```

# 8.6 与标准检索进行比较
实现一种标准的检索方法，以便与RSE（相关段落提取）进行比较：


```python
def standard_top_k_retrieval(pdf_path, query, k=10, chunk_size=800):
    """
    标准RAG（检索增强生成）方法，基于Top-K检索。

    Args:
        pdf_path (str): 文档路径
        query (str): 用户查询
        k (int): 需要检索的块数量
        chunk_size (int): 每个块的大小

    Returns:
        Dict: 包含查询、检索到的块和响应的结果字典
    """
    print("\n=== 开始标准TOP-K检索 ===")
    print(f"查询: {query}")

    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)

    # 为查询创建嵌入
    print("创建查询嵌入并检索块...")
    query_embedding = create_embeddings([query])[0]

    # 基于查询嵌入检索最相关的前k个块
    results = vector_store.search(query_embedding, top_k=k)
    retrieved_chunks = [result["document"] for result in results]

    # 将检索到的块格式化为上下文字符串
    # context = "\n\n".join([
    #     f"CHUNK {i+1}:\n{chunk}"
    #     for i, chunk in enumerate(retrieved_chunks)
    # ])
    context = "\n\n".join([
        f"文本块 {i+1}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    # 使用上下文从语言模型生成响应
    response = generate_response(query, context)

    # 将结果编译成字典
    result = {
        "query": query,  # 用户查询
        "chunks": retrieved_chunks,  # 检索到的块
        "response": response  # 模型生成的响应
    }

    print("\n=== 最终响应 ===")
    print(response)

    return result

```

# 8.7 评估 RSE


```python
def evaluate_methods(pdf_path, query, reference_answer=None):
    """
    比较RSE（相关段落提取）与标准Top-K检索方法。

    Args:
        pdf_path (str): 文档路径
        query (str): 用户查询
        reference_answer (str, 可选): 用于评估的参考答案
    """
    print("\n========= 评估开始 =========\n")

    # 运行带有RSE（相关段落提取）的RAG方法
    rse_result = rag_with_rse(pdf_path, query)

    # 运行标准Top-K检索方法
    standard_result = standard_top_k_retrieval(pdf_path, query)

    # 如果提供了参考答案，则评估响应结果
    if reference_answer:
        print("\n=== 比较结果 ===")

        # 创建一个评估提示，将两种响应与参考答案进行比较
        evaluation_prompt = f"""
            查询：{query}

            参考答案：
            {reference_answer}

            标准检索系统回答内容：
            {standard_result["response"]}

            相关片段提取系统回答内容：
            {rse_result["response"]}

            请基于以下维度对比两个系统的回答内容质量：
            1. 准确性与完整性
            2. 问题解决有效性
            3. 信息冗余度

            具体要求：
            • 判断哪个系统在各维度表现更优
            • 对比分析时须直接引用具体内容
            • 请逐项说明判断依据
        """

        print("正在根据参考答案评估回答...")

        # 使用指定模型生成评估结果
        evaluation = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "您是RAG系统回答内容的客观评估专家。"},
                {"role": "user", "content": evaluation_prompt}
            ]
        )

        # 打印评估结果
        print("\n=== 评估结果 ===")
        print(evaluation.choices[0].message.content)

    # 返回两种方法的结果
    return {
        "rse_result": rse_result,
        "standard_result": standard_result
    }

```


```python
# Load the validation data from a JSON file
with open('../../data/val.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Extract the reference answer from the validation data
reference_answer = data[0]['ideal_answer']

# Run evaluation
results = evaluate_methods(pdf_path, query, reference_answer)
```

    
    ========= 评估开始 =========
    
    
    === 开始带有相关段落提取的RAG ===
    查询: 什么是‘可解释人工智能’，为什么它被认为很重要？
    从文档中提取文本...
    将文本切分为非重叠段落...
    创建了 13 个块
    为块生成嵌入向量...
    
    计算相关性分数和块值...
    寻找最佳连续文本段落...
    找到段落 (0, 13)，得分 7.6610
    
    从块中重建文本段落...
    正在使用相关段落作为上下文生成响应...
    
    === 最终响应 ===
    可解释人工智能（XAI，Explainable Artificial Intelligence）是指通过技术手段使人工智能系统的决策过程变得透明、可理解和可追溯的领域。其核心目标是解决传统AI模型（尤其是深度学习等复杂模型）的"黑箱"问题，让人类能够理解AI如何得出特定结论或做出特定决策。
    
    重要性体现在以下关键方面：
    
    1. 信任建立（第20章）
    - 透明度是建立用户对AI系统信任的基础
    - 可解释性让用户能够评估AI决策的可靠性和公平性
    
    2. 伦理合规（第19章）
    - 满足道德AI原则中的透明度和问责要求
    - 支持"负责任AI"的开发框架（第21章）
    - 是AI治理和监管的核心要求（第18章）
    
    3. 实际应用需求
    - 在医疗诊断（第11章）、金融风控（第7章）等高风险领域，决策依据必须可验证
    - 帮助发现和修正模型偏见（第4章提及的偏见放大问题）
    - 支持人机协作场景中的决策协调（第8章工作场景应用）
    
    4. 技术发展层面（第15章）
    - 是AI研究的前沿方向之一
    - 与"以人为中心的AI"发展理念深度契合
    - 促进更健壮、可靠的AI系统开发
    
    典型技术包括：
    - 决策可视化工具
    - 特征重要性分析
    - 局部可解释模型（LIME）
    - 反事实解释方法等
    
    随着AI在关键领域应用的深化（如第14章智慧城市、第12章网络安全），可解释性已成为确保AI系统安全、可靠、公平的必要条件，也是平衡技术创新与社会接受度的关键因素。
    
    === 开始标准TOP-K检索 ===
    查询: 什么是‘可解释人工智能’，为什么它被认为很重要？
    从文档中提取文本...
    将文本切分为非重叠段落...
    创建了 13 个块
    为块生成嵌入向量...
    创建查询嵌入并检索块...
    正在使用相关段落作为上下文生成响应...
    
    === 最终响应 ===
    **可解释人工智能（XAI）**是指通过技术手段使人工智能系统的决策过程透明化、易于理解的人工智能分支。其核心目标是破解AI模型的"黑箱"特性，让人类能够追溯和理解算法如何得出特定结论。
    
    **重要性主要体现在以下方面**：
    
    1. **信任建立**（文本块3/6/9）
       - 当医疗诊断、金融风控等关键领域采用AI时，决策透明性直接影响用户信任度。XAI通过展示决策逻辑（如特征重要性分析、决策路径可视化）增强系统可信度。
    
    2. **伦理与责任**（文本块1/3/4）
       - 涉及偏见的算法可能放大社会歧视（如招聘AI中的性别偏见）。XAI技术能识别偏差来源（文本块3），为算法审计提供依据，满足欧盟《AI法案》等合规要求。
    
    3. **安全验证**（文本块5/9）
       - 在自动驾驶等高风险场景中，XAI可验证系统是否基于合理特征（如交通标志而非无关像素）做出判断，避免因数据噪声导致的致命错误。
    
    4. **性能优化**（文本块6）
       - 可解释性分析能暴露模型弱点（如过度依赖某个非稳健特征），帮助开发者改进算法。深度学习领域特别关注通过注意力机制等XAI技术提升模型鲁棒性。
    
    5. **人机协作**（文本块1/7）
       - 当AI作为人类决策辅助工具时（如医疗诊断），医生需要理解AI建议的依据以做出最终判断，XAI是实现有效协同的基础。
    
    当前主要技术包括LIME（局部解释）、SHAP值（特征贡献度量化）以及针对神经网络的层间可视化等。随着欧盟《数字服务法案》等法规强化AI透明度要求（文本块9），XAI已成为负责任AI开发的必备组件。
    
    === 比较结果 ===
    正在根据参考答案评估回答...
    
    === 评估结果 ===
    ### 1. 准确性与完整性  
    **标准检索系统更优**  
    - **准确性**：标准检索系统提供了具体的技术示例（如LIME、SHAP值、注意力机制）和实际应用场景（医疗诊断、金融风控、自动驾驶），这些内容与XAI的核心定义和重要性高度吻合。例如，提到"通过特征重要性分析、决策路径可视化增强信任"（文本块3/6/9），直接关联XAI的技术实现。  
    - **完整性**：覆盖了信任、伦理、安全、性能优化、人机协作五大维度，并引用法规（如欧盟《AI法案》）和具体技术（层间可视化），信息全面。相比之下，相关片段系统的"伦理合规"部分仅泛泛提及"道德AI原则"（第19章），缺乏具体技术或案例支撑。  
    
    **引用对比**：  
    - 标准检索系统：*"XAI技术能识别偏差来源（文本块3），为算法审计提供依据，满足欧盟《AI法案》等合规要求"*（具体法规和技术结合）。  
    - 相关片段系统：*"满足道德AI原则中的透明度和问责要求"*（未说明如何满足）。  
    
    ---
    
    ### 2. 问题解决有效性  
    **标准检索系统更优**  
    - **针对性**：标准检索系统直接回答"为什么重要"，每个重要性论点均配以实际应用场景和技术手段。例如，针对"安全验证"提到"避免因数据噪声导致的致命错误"（文本块5/9），问题解决路径清晰。  
    - **实用性**：相关片段系统虽列出四大重要性（信任、伦理、应用需求、技术发展），但部分内容重复（如"伦理合规"与"负责任AI"框架），且未明确技术如何解决问题。例如，第4章提及"偏见放大问题"但未说明XAI如何修正。  
    
    **引用对比**：  
    - 标准检索系统：*"可解释性分析能暴露模型弱点（如过度依赖某个非稳健特征），帮助开发者改进算法"*（具体问题+解决方案）。  
    - 相关片段系统：*"帮助发现和修正模型偏见"*（未说明修正方法）。  
    
    ---
    
    ### 3. 信息冗余度  
    **相关片段系统更优**  
    - **简洁性**：相关片段系统通过章节引用（如第11章医疗诊断、第7章金融风控）概括领域应用，避免了过多技术细节，冗余度低。标准检索系统虽信息全面，但部分内容重复（如"信任建立"在文本块3/6/9中多次出现）。  
    - **结构化**：相关片段系统用分点（1. 信任建立，2. 伦理合规等）清晰划分重要性，而标准检索系统的"性能优化"与"安全验证"部分内容有重叠（均涉及模型可靠性）。  
    
    **引用对比**：  
    - 相关片段系统：*"典型技术包括：决策可视化工具、特征重要性分析"*（仅列举技术名称）。  
    - 标准检索系统：*"当前主要技术包括LIME（局部解释）、SHAP值（特征贡献度量化）以及针对神经网络的层间可视化等"*（详细解释技术，可能超出必要信息）。  
    
    ---
    
    ### 总结  
    - **标准检索系统**在**准确性**和**问题解决有效性**上显著胜出，因其提供具体技术、法规和场景化解释。  
    - **相关片段系统**在**信息冗余度**上更优，结构简洁但牺牲了部分深度。  
    - 若用户需要专业解答（如技术人员），推荐标准检索系统；若需快速概览，相关片段系统更合适。

