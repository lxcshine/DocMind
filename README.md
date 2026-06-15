# DocMind

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.0.0-blue" alt="Version" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Backend-FastAPI%20%2B%20RAG--Anything-1677ff" alt="Backend" />
  <img src="https://img.shields.io/badge/Frontend-React%20%2B%20Ant%20Design%20v5-1677ff" alt="Frontend" />
  <img src="https://img.shields.io/badge/RAG-LightRAG%20%2B%20MinerU-1677ff" alt="RAG" />
  <img src="https://img.shields.io/badge/MCP-Streamable%20HTTP-1677ff" alt="MCP" />
</p>

<h1 align="center">DocMind</h1>

<p align="center">
  <strong>企业级文档智能平台</strong>
  <br />
  <em>将非结构化文档转化为可查询、可对话、可记忆的知识引擎</em>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> •
  <a href="#rag-引擎">RAG 引擎</a> •
  <a href="#mcp-集成">MCP 集成</a> •
  <a href="#项目结构">项目结构</a> •
  <a href="#api-参考">API 参考</a> •
  <a href="#部署指南">部署指南</a>
</p>

---

## 为什么选择 DocMind

传统 RAG 方案在生产环境中面临的核心痛点：

| 痛点 | 传统方案 | DocMind 解法 |
|------|---------|-------------|
| **50 页 PDF 处理 30 分钟** | 全量 KG 提取，每 chunk 一次 LLM 调用 | **三档处理模式**：Fast（1-2 分钟）/ Standard（3-8 分钟）/ Full（10-30 分钟），按需选择 |
| **200 文件批量导入 OOM** | 串行处理，内存爆炸 | **BatchProcessor**：`asyncio.Semaphore(3)` 并发控制，单文档进度追踪，优雅降级 |
| **Agent 50 轮后忘记指令** | 简单截断 + flat summary | **ContextManager v2**：分层摘要压缩 + 指令持久化 + 动态预算分配 + 去重守卫 |
| **只有一种检索模式** | 单一向量相似度 | **LightRAG 五模式查询**：local / global / hybrid / mix / naive + RF-Mem 自适应切换 |
| **Windows 长路径崩溃** | MinerU 直接处理原始路径 | **短路径代理**：MD5 哈希生成 ≤8 字符临时文件名，绕过 MAX_PATH 限制 |
| **UI 一股 AI 模板味** | 默认 Ant Design 蓝白配色 | **Vercel/Linear 风格设计系统**：CSS 变量体系 + 深色空间调色板 + 毛玻璃表面 |


### 项目功能

![Homepage Screenshot](https://github.com/lxcshine/DocMind/blob/master/image/image_1.png)

![Homepage Screenshot](https://github.com/lxcshine/DocMind/blob/master/image/image_2.png)

![Homepage Screenshot](https://github.com/lxcshine/DocMind/blob/master/image/image_3.png)

![Homepage Screenshot](https://github.com/lxcshine/DocMind/blob/master/image/image_4.png)

![Homepage Screenshot](https://github.com/lxcshine/DocMind/blob/master/image/image_5.png)

---

## RAG 引擎

DocMind 的 RAG 管线采用**分层**设计：处理模式（fast/standard/full）、查询模式（5 种）、自适应/检索增强（RF-Mem）、Agentic 检索（PageIndex 风格）——每个环节都可独立配置。

### 三档处理模式

| 模式 | 管线 | 单文档耗时 | 适用场景 |
|------|------|-----------|---------|
| **Fast** | MinerU 解析 → LightRAG 分块 → Embedding → 向量 upsert（**无 KG**） | 1-2 分钟 | 快速检索，不需要关系推理 |
| **Standard** | MinerU 解析 → LightRAG `ainsert`（KG + 向量，默认 chunk 大小） | 3-8 分钟 | 通用场景，实体关系图谱 + 向量检索 |
| **Full** | MinerU 解析 → KG + 向量 + **多模态**（图片/表格/公式 VLM 分析） | 10-30 分钟 | 文档含大量图表、公式，需要 VLM 深度理解 |

**Fast 模式底层实现**（`core/raganything.py`）：

```python
# 绕过 ainsert() 的 KG 阶段，直接操作 LightRAG 的存储层
chunks       = lightrag.chunking(text_content)            # 1. 分块
embeddings   = await lightrag.embedding_func(chunks)      # 2. Embedding（1 次 API 调用）
for chunk, vec in zip(chunks, embeddings):                 # 3. 批量 upsert（无 KG）
    await lightrag.chunks_vdb.upsert(chunk_id, vec, {...})
```

### 五模式查询

LightRAG 提供五种查询策略，前端可通过下拉菜单切换：

| 模式 | 原理 | 适用问题 |
|------|------|---------|
| `naive` | 纯向量相似度，返回 top-K chunks | "文档中 X 的原文是什么" |
| `local` | 实体召回 → 局部子图遍历 | "跟这个实体相关的有哪些" |
| `global` | 全图摘要，宏观视角 | "帮我总结一下 Y 领域的整体情况" |
| `hybrid` | local + global 加权融合 | 需要兼顾细节和全局 |
| `mix`（默认） | local/global/naive 动态选择 | 通用 Q&A |

### 自适应检索（RF-Mem 启发）

实现于 `core/adaptive_retrieval.py`。核心思想：不是所有查询都需要图检索——**熟悉的查询用向量就够了，陌生的查询才需要图探索**。

**双路径策略：**

1. **Probe（探测）**：低成本 top-K=5 向量召回
2. **Familiarity 评估**：基于探测结果的 `mean_score`（平均相似度）和 `entropy`（确定性）
3. **自适应切换**：
   - `familiarity ≥ 0.65` 且 `entropy ≤ 1.5` → **Familiarity 路径**：直接用向量结果，不做图遍历
   - 否则 → **Recollection 路径**：启动图探索，多轮链式回忆（聚类 + alpha 混合 + 迭代扩展）

### Agentic 检索（PageIndex 风格）

实现于 `core/agentic_retrieve.py`。传统 RAG 只能给"页面"级答案，但用户要的是"章节"级理解。

1. 文档入库时构建**层级结构树**：章节 → 子章节 → 页面
2. 暴露三个工具给 LLM Agent：
   - `get_document(doc_id)` → 元数据
   - `get_document_structure(doc_id)` → 层级结构（无文本，省 token）
   - `get_page_content(doc_id, pages)` → 指定页面原文
3. Agent 自主决定调用顺序，直到获取足够信息后生成摘要

### 上下文管理（ContextManager v2）

解决多轮对话中 Agent 忘记早期指令、重复已给答案的问题。四大机制：

| 机制 | 原理 | 解决的问题 |
|------|------|-----------|
| **分层摘要压缩** | Level 0 原始消息 → Level 1 Chunk 摘要（每 6 轮）→ Level 2 Session 摘要 | 早期信息丢失 |
| **指令持久化** | System prompt 永不压缩 + 关键规则（MUST/NEVER/必须/绝不）在高注意力位置重复注入 | 指令稀释 |
| **动态预算分配** | 浅层对话 60% 逐字历史 → 深层对话 40% 逐字 + 20% 摘要，信息密度恒定 | 128k 窗口反而更差 |
| **去重守卫** | Jaccard 相似度检测重复问题 + 注入 `[DO NOT repeat]` 提示 | 重复回答 |

上下文采用**三明治结构**缓解 Lost-in-the-Middle 效应：关键信息（指令/摘要/去重提示）放在首尾，细节放中间。

---

## MCP 集成

DocMind 内置 MCP（Model Context Protocol）Server，将核心能力暴露为标准化工具，任何 MCP 兼容客户端（Claude Desktop、Cursor、VS Code Copilot 等）可直接调用。

### 暴露的能力

**7 个 Tools（AI 可调用的函数）：**

| Tool | 作用 | 底层实现 |
|------|------|---------|
| `search_knowledge_base` | RAG 知识库搜索（5 种模式） | `raganything.query()` |
| `get_document` | 获取文档元数据 | `tree_index_store.get_document_info()` |
| `get_document_structure` | 获取文档层级结构 | `tree_index_store.get_document_structure()` |
| `get_page_content` | 获取指定页面内容 | `tree_index_store.get_page_content()` |
| `search_web` | 深度网页搜索 + LLM 综合 | `WebScraper` + LLM |
| `memory_add` | 添加记忆条目 | `MemoryService.add_entry()` |
| `memory_search` | 语义搜索记忆 | `MemoryService._index.search()` |

**2 个 Resources（AI 可读取的数据）：**

| Resource URI | 作用 |
|-------------|------|
| `document://list` | 知识库文档列表 |
| `document://{doc_id}/structure` | 文档层级结构（模板化） |

**1 个 Prompt（预定义模板）：**

| Prompt | 作用 |
|--------|------|
| `knowledge_qa` | 知识库问答 Prompt 模板 |

### 传输协议

采用 **Streamable HTTP**（MCP 规范 2025-11-25）：
- 单一 `/mcp` 端点，POST 发送 JSON-RPC，GET/SSE 支持流式响应
- `stateless_http=True` 无状态模式，支持负载均衡器水平扩展
- 兼容反向代理（nginx、云 LB）

### 客户端配置

**Claude Desktop** (`settings.json`)：
```json
{
  "mcpServers": {
    "docmind": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

**Cursor / VS Code**：
```json
{
  "mcp.servers": {
    "docmind": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

---

## 快速开始

### 环境要求

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.11+ | 后端运行时 |
| Node.js | 18+ | 前端构建 |
| MySQL | 8.0+ | 会话历史、文档元数据 |
| Tesseract | 5.0+ | OCR 功能（可选） |

### 1. 克隆项目

```bash
git clone https://github.com/lxcshine/DocMind.git
cd DocMind
```

### 2. 后端配置

```bash
cd backend

# 创建虚拟环境
conda create -n docmind python=3.11
conda activate docmind

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填写以下必填项：
#   GEMINI_API_KEY=your_api_key
#   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
#   GEMINI_MODEL=gemini-2.5-flash
#   MYSQL_HOST=localhost
#   MYSQL_PASSWORD=your_password
```

### 3. 启动后端

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 5. 访问

- 前端：http://localhost:5173
- 后端 API 文档：http://localhost:8000/api/docs
- MCP 端点：http://localhost:8000/mcp

---

## 5. Agent 如何阅读文档

这一节回答一个根本问题：**当用户上传了一篇论文并开始提问时，Agent 在背后做了什么？** 理解这个过程，也就理解了 DocMind 与传统 RAG 在范式上的区别。

### 5.1 以页面为单元，不做 chunk 切分

考虑一篇归一化方法论文中的典型段落：

```
3.2 Root Mean Square Layer Normalization

RMS(x) = sqrt(1/n * Σ_{i=1}^{n} x_i^2)                    (1)

Unlike LayerNorm which centers and scales the inputs,
RMSNorm removes only the scale invariance...

∂L/∂x_i = (g_i / RMS(x)) * (∂L/∂x̂_i - (x̂_i / n) * Σ_{j=1}^{n} x̂_j * ∂L/∂x̂_j)   (2)
```

如果按 512 token 切分，(1) 和 (2) 有很大概率被分配到不同的 chunk 中。当 LLM 被问到"RMSNorm 的梯度公式是什么"时，它收到的 chunk 可能只包含梯度推导的后半部分，上下文完全断裂。

更深层的问题是：embedding 模型对公式的语义理解几乎为零。`∂L/∂x_i` 和 `∂L/∂w` 在向量空间的余弦相似度可能高达 0.95，因为它们的上下文词汇高度重叠——但前者讨论的是输入梯度，后者讨论的是权重梯度，数学含义截然不同。

DocMind 采取的策略简单而直接：**以物理页面为最小检索单元。** 一页就是一块完整的上下文，不做切割。Agent 每次翻页时看到的是完整的页面内容，无论上面是纯文本、推导公式还是对比表格。

### 5.2 建索引：从 PDF 书签到层级树

Agent 阅读文档的第一步不是向量化，而是建立一张"地图"。

PDF 规范本身支持书签（Outline）——作者在撰写论文时已经划分好的章节结构。PyMuPDF 可以直接读取这些书签，无需调用 LLM：

```
PDF 书签                                Agent 看到的地图
                                       
1. Introduction                        RMSNorm.pdf (8 pages)
  1.1 Background                      ├── 1. Introduction               [p1-2]
  1.2 Related Work                    │   ├── 1.1 Background            [p1]
2. Method                             │   └── 1.2 Related Work          [p2]
  2.1 Definition                      ├── 2. Method                     [p3-5]
  2.2 Gradient Analysis               │   ├── 2.1 Definition            [p3]
  2.3 Implementation                  │   ├── 2.2 Gradient Analysis     [p4]
3. Experiments                        │   └── 2.3 Implementation        [p5]
  3.1 Setup                           └── 3. Experiments                [p6-8]
  3.2 Results                             ├── 3.1 Setup                 [p6]
                                          └── 3.2 Results               [p7-8]
```

建索引的全过程：

1. **提取书签** — PyMuPDF 从 PDF 中读取原始 Outline 数据，得到完整的层级目录。
2. **计算页码范围** — 递归计算每个节点的 `page_from` 和 `page_to`，精确标注每节占据的物理页面。
3. **逐页提取文本** — 将每页的文本内容独立存储为 `.txt` 文件。
4. **渲染页面图片** — 以 2 倍 DPI 渲染每页为图片，供视觉模型按需调用。
5. **生成摘要（可选）** — 对每个节点调用 LLM 生成一句话描述，存入树节点的 `summary` 字段，帮助 Agent 快速判断相关性。

整个过程对没有书签的 PDF 也是可工作的——此时整篇文档作为一个平铺的页面列表，Agent 仍然可以逐页翻阅，只是缺少了按章节定位的便利。

### 5.3 检索：Agent 自主翻页

这是 DocMind 与传统 RAG 最本质的分界线。

传统 RAG 的检索是一个搜索引擎：输入 query，输出 top-k 个"最相似"的结果。这个流程是固定的——query → embedding → ANN search → top-k——没有中间环节，没有决策空间。

DocMind 的检索是一个 **Agent 的多轮自主决策过程**。Agent 拿到目录树（地图）后，根据问题的复杂度和自己的判断，分多轮翻阅文档：

```
用户提问："对比 RMSNorm、LayerNorm 和 BatchNorm 的核心公式差异"

Round 1 — Agent 决定：先了解三篇文档的基本信息
  get_document("rms_id"), get_document("ln_id"), get_document("bn_id")

Round 2 — Agent 决定：查看三篇文档的目录，找到公式所在的章节
  get_document_structure("rms_id")  → 发现 2.1 节含公式，在 p3
  get_document_structure("ln_id")   → 发现 2. Method 含公式，在 p2-4
  get_document_structure("bn_id")   → 发现 3. Algorithm 含公式，在 p4-5

Round 3 — Agent 决定：翻阅这三篇文档中与公式相关的页面
  get_page_content("rms_id", "3")
  get_page_content("ln_id", "2-4")
  get_page_content("bn_id", "4-5")

Round 4 — 综合三篇文档的信息，生成对比回答
```

Agent 的每一步工具调用都是它基于当前已获取的信息自主决策的结果。如果第二轮发现某篇文档不相关，它完全可以停止翻阅。如果第三轮发现还需要更多页面，它会继续翻。这个灵活性是固定的 top-k 搜索无法提供的。

### 5.4 容错机制

当 LLM API 不可用时（余额不足、网络故障），Agent 自动降级为关键词匹配模式：对用户查询词和所有页面文本做 TF 词频匹配，按相关度排序后返回 top-k 页面，用检索到的内容构造 prompt 直接回答。虽然不如 Agent 自主检索精准，但保证了核心功能的可用性。

---

## 6. 知识库隔离

当一个知识库中同时存在多篇文档时，隔离是 Agent 必须解决的问题。DocMind 的知识库隔离不是靠多出来的某个"隔离模块"实现的，而是其索引设计的自然结果。

### 6.1 文件系统即隔离边界

每篇文档在 `tree_index/` 目录下拥有完全独立的存储空间：

```
tree_index/
├── _meta.json                    # 全局元数据索引
├── abc123.json                   # RMSNorm 的目录树
├── abc123_pages/                 # RMSNorm 的逐页文本
│   ├── 1.txt
│   └── 2.txt
├── abc123_images/                # RMSNorm 的逐页图片
│   ├── 1.jpg
│   └── 2.jpg
├── def456.json                   # LayerNorm 的目录树
├── def456_pages/                 # LayerNorm 的逐页文本
└── def456_images/                # LayerNorm 的逐页图片
```

组件之间通过 `doc_id` 访问：

- `tree_index.py` — 只能通过 `doc_id` 读取特定文档的树结构和页面内容。
- `agentic_retrieve.py` — 每个工具调用都携带明确的 `doc_id`，Agent 不可能在翻阅 RMSNorm 时意外读到 LayerNorm 的内容。
- `chat.py` — 每条回答末尾标注信息来源的文档名和页码范围。

三层隔离中，最关键的是第二层：Agent 的工具调用机制天然保证了隔离。工具签名 `get_page_content(doc_id, pages)` 中的 `doc_id` 是必填参数，不存在"所有文档混在一起搜"的模式。

### 6.2 跨文档不混淆

当用户同时提问多篇文档时，Agent 的行为如下：

```
知识库中有 RMSNorm、LayerNorm、BatchNorm 三篇文档

用户问题："RMSNorm 和 LayerNorm 的区别是什么？"

Agent 行为：
  - 翻阅 RMSNorm 的目录和页面（doc_id = "rms_id"）
  - 翻阅 LayerNorm 的目录和页面（doc_id = "ln_id"）
  - 自主判断 BatchNorm 与当前问题无关，不翻阅
  - 最终回答中不包含任何 BatchNorm 的内容
```

Agent 既能在必要时跨文档检索，又不会把不相关文档的内容混入回答。

---

## 7. 设计决策：检索策略与索引策略

前文描述了 DocMind"怎么做"，这一节解释"为什么这样做"。每一个设计决策都对应着传统方案中的一个具体痛点。

### 7.1 为什么以页面为单元

**对应痛点：chunk 切分破坏语义完整性。**

学术文档的信息结构天然以页面为单元。一页论文通常包含一个完整的论证段落、一个公式群、或一张对比表格。在这个粒度上，语义是自包含的。

页面作为检索单元还有一个工程优势：它和 PDF 的物理结构一致，不需要任何参数调优。没有 chunk size、没有 overlap ratio、没有分离策略——需要调参这件事本身就是方法不鲁棒的信号。

### 7.2 为什么用目录树而非向量索引

**对应痛点：embedding 模型理解不了专业内容。**

目录是论文作者亲手写的。它反映了作者对文档逻辑结构的组织意图，远比 embedding 模型对文本的"理解"可靠。当 Agent 看到目录中"2.1 RMSNorm Definition"这个节点时，它不需要做任何语义匹配就知道——如果用户问的是 RMSNorm 的定义，答案大概率在 p3。

人类研究者翻论文的第一步永远是看目录。我们不过是在让 Agent 做同样的事情。

### 7.3 为什么用 Agent 工具调用而非固定检索

**对应痛点：固定检索逻辑无法应对复杂信息需求。**

考虑这样一个查询："对比这三篇论文中关于归一化方法的理论推导部分，重点看梯度公式的差异。"这个查询里包含了三个意图：多文档对比、特定章节定位、特定内容类型过滤。传统 RAG 的 query → top-k 流程无法区分"理论推导"和"实验结果"，也做不到跨文档逐章节对比。

Agent 的多轮工具调用天然适用于这种场景。它把复杂的检索需求分解为一系列简单的操作——先找结构，再翻页面，最后综合——每一步都是可解释的。

### 7.4 为什么用视觉模型而非 OCR

**对应痛点：文本提取破坏公式和表格的排版。**

从 PDF 提取文本时，`\sum_{i=1}^{n}` 可能变成 `i=1n`，表格的列对齐完全消失。OCR 虽然能识别字符，但对数学符号的排版结构同样无能为力。

让视觉模型直接"看"页面图片，得到的输出是 `$$\sum_{i=1}^{n}$$`——LaTeX 格式，完美保留排版。代价是视觉模型调用更贵、更慢，所以只在检测到公式或表格时才切换。

---

## 8. 上下文记忆

对话式 AI 的一个经典困境：用户在三轮对话前提到自己"主要关注归一化方法的理论推导"，到第四轮 Agent 已经忘了。传统做法是把历史对话全部塞进 prompt，但 token 窗口有限，且用户真正需要跨越对话记住的只是少数关键信息。

DocMind 的记忆系统解决这个问题的方式是：**不是把对话历史全塞进去，而是从历史中提取值得记住的东西，再用关键词检索精准注入。**

### 8.1 双层记忆架构

```
┌─────────────────────────────────────────────────────────┐
│                    会话记忆层（MySQL）                    │
│  当前会话的完整对话历史，按时间顺序存储                    │
│  作用：支持 Agent 回顾当前会话中的上下文                   │
│  生命周期：会话级，用户可手动删除会话                      │
└──────────────────────────┬──────────────────────────────┘
                           │ 对话结束时触发
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   长期记忆层（JSON + 关键词索引）          │
│  从对话中提取的结构化记忆，跨会话持久化                    │
│  作用：让 Agent 记住用户的兴趣、偏好和历史决策             │
│  生命周期：持久化，FIFO 策略管理容量                      │
└─────────────────────────────────────────────────────────┘
```

### 8.2 四种记忆类型

当一轮对话结束时，记忆模块会调用 LLM 从对话中提取结构化的记忆条目：

- **Raw Memory** — 原始对话的完整文本，不做任何处理。在检索时权重最低，仅在关键词高度匹配时被召回。
- **Semantic Memory** — 用户表达的知识偏好和兴趣方向。例如"用户在对比归一化方法时更关注理论推导而非实验性能"会被提取为一条 Semantic Memory，附带关键词 `[归一化, 理论推导, 数学证明]`。
- **Episodic Memory** — 对话中的关键事件。例如"2024-06-01 用户对比了 RMSNorm、LayerNorm 和 BatchNorm 的公式差异"。
- **Procedural Memory** — 用户的交互习惯。例如"用户偏好 LaTeX 格式的公式输出"、"用户希望回答保持学术严谨风格"。

每种记忆类型的提取由 LLM 根据 System Prompt 中的定义完成，输出格式为 JSON，包含 `content`（记忆内容）和 `keywords`（关键词列表）两个字段。

### 8.3 记忆的检索与注入

当用户发起新对话时，记忆模块执行以下流程：

1. **关键词匹配** — 从用户消息中提取关键词，与所有记忆条目的关键词索引做交集匹配。
2. **排序与截断** — 按匹配度排序，保留 top-5 条最相关的记忆。
3. **注入 System Prompt** — 将选中的记忆条目格式化后注入 Agent 的 System Prompt：

```
[记忆上下文]
以下是从您与用户的历史对话中提取的相关记忆，请参考这些信息：
- [Semantic] 用户在对比归一化方法时更关注理论推导而非实验性能
- [Episodic] 上次对话中用户对比了 RMSNorm 和 LayerNorm
- [Procedural] 用户偏好 LaTeX 格式的公式输出
```

Agent 在生成回答时会自然地将这些记忆融入回复——比如自动使用 LaTeX 格式渲染公式，优先从理论推导的角度组织对比分析。

### 8.4 容量管理

长期记忆采用 FIFO（先进先出）策略管理容量。当记忆条目总数超过上限时，最早创建的条目被自动删除。四种记忆类型的存储配额可独立配置，保证用户偏好类记忆比原始对话日志有更长的保留周期。

---

## 9. 致谢

DocMind 的设计理念深受 [PageIndex](https://github.com/nick-cao/PageIndex) 项目 Vectorless RAG 思想的启发，记忆模块借鉴了 [RAGFlow](https://github.com/infiniflow/ragflow) 的多类型记忆设计。感谢这些项目的开创性工作。
