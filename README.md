# DocMind

> 推理驱动的智能文档知识库系统 | Reasoning-Driven Document Knowledge Base

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178c6.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ? 1. 项目简介

**DocMind** 是一个基于 **推理驱动 + 无向量（Vectorless RAG）** 范式的智能文档知识库系统。与传统的向量检索 RAG 不同，DocMind 借鉴了 **PageIndex** 的核心理念：**不做 chunk 切分，不做向量嵌入，由 LLM Agent 自主导航文档的层级树结构来精准定位和检索内容。**

核心哲学：**让 LLM 像一个真正的研究者那样"翻阅"文档——先看目录（层级树骨架），定位到相关章节，再翻开对应页面阅读，最后综合所有信息给出答案。** 整个过程无需向量数据库、无需 embedding 模型、无需 chunk 策略调优。

> 项目原名 ResearchFlow，现已更名为 DocMind。

---

## ? 2. 项目功能

### 2.1 核心功能矩阵

| 功能模块 | 描述 |
|---------|------|
| **? 知识库管理** | 上传 PDF / Markdown / TXT 文档，一键加入知识库，实时进度追踪 |
| **? 智能问答** | 基于 Agentic RAG 的文档问答，支持流式输出 (token-by-token) |
| **? 智能检索** | LLM Agent 自主调用工具导航文档树，精准定位内容 |
| **?? Vision RAG** | 公式 / 表格页面自动走视觉模型，直接读取图片中的公式和表格排版 |
| **? Memory 记忆** | 4 类记忆体系（原始/语义/情景/程序性），自动从对话中提取知识 |
| **? OCR 识别** | Tesseract + LLM 智能纠错，支持 PDF / 图片 / PPT 多格式 |
| **? 网页抓取** | Firecrawl 风格多引擎网页内容抓取，输出 Markdown |
| **? 对话持久化** | MySQL 存储聊天历史，支持会话管理 |
| **? 可控生成** | 支持 Stop 按钮打断 LLM 生成，任务取消与超时保护 |

### 2.2 双模型架构

```
文本模型 (GLM)  ←→  纯文本问答、文档解析、对话生成  ← 成本低
视觉模型 (VLM)  ←→  公式渲染、表格解析、图表理解  ← 按需调用
```

系统通过正则检测自动判断是否需要视觉模型，仅在内容包含公式或表格时启用，最大限度节省 API 费用。

---

## ? 3. 项目结构

```
DocMind/
├── backend/                          # Python 后端 (FastAPI)
│   ├── main.py                       # 应用入口，路由注册
│   ├── requirements.txt              # Python 依赖
│   ├── .env.example                  # 环境变量模板
│   ├── config/
│   │   └── settings.py               # 全局配置类，环境变量管理
│   ├── api/                          # API 路由层
│   │   ├── documents.py              # 文档上传/处理/删除/列表 API
│   │   ├── chat.py                   # 核心问答 API (/chat, /stream)
│   │   ├── search.py                 # 智能检索 API
│   │   ├── memory.py                 # 记忆管理 API
│   │   └── ocr.py                    # OCR 识别 API
│   ├── core/                         # 核心业务逻辑层
│   │   ├── parser.py                 # PageIndex 风格文档解析器
│   │   ├── doc_handler.py            # 文档处理流水线（上传→解析→建树→存储）
│   │   ├── agentic_retrieve.py       # Agentic 检索器（LLM 工具调用）
│   │   ├── tree_index.py             # 层级树索引存储（无向量）
│   │   ├── meta_store.py             # 文档元数据管理
│   │   ├── progress.py               # 实时处理进度追踪
│   │   ├── chat_history.py           # MySQL 对话历史存储
│   │   ├── memory.py                 # RAGFlow 风格记忆服务
│   │   ├── ocr_handler.py            # 智能 OCR 处理器
│   │   └── web_scraper.py            # Firecrawl 风格网页抓取器
│   ├── tree_index/                   # 树索引持久化目录
│   ├── uploads/                      # 上传文件存储目录
│   └── tests/                        # 测试文件
├── frontend/                         # React 前端 (Vite + TypeScript + Ant Design)
│   ├── package.json                  # Node.js 依赖
│   ├── vite.config.ts                # Vite 构建配置
│   ├── src/
│   │   ├── App.tsx                   # 路由配置
│   │   ├── main.tsx                  # 前端入口
│   │   ├── layouts/
│   │   │   └── MainLayout.tsx        # 主布局（侧边栏导航）
│   │   ├── pages/
│   │   │   ├── KnowledgeBase/        # 知识库页面
│   │   │   ├── Chat/                 # 智能问答页面
│   │   │   ├── Search/               # 智能检索页面
│   │   │   ├── Memory/               # 记忆管理页面
│   │   │   └── OCR/                  # OCR 识别页面
│   │   └── styles/
│   │       └── pages.css             # 全局样式
│   └── index.html
└── README.md
```

---

## ?? 4. 项目安装与使用

### 4.1 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| MySQL | 5.7+ (可选，对话持久化) |
| Tesseract OCR | 5.x (可选，OCR 功能) |

### 4.2 后端安装

```bash
# 1. 进入后端目录
cd backend

# 2. 创建虚拟环境（推荐）
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key

# 6. 启动后端服务
python main.py
# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4.3 前端安装

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev

# 4. 构建生产版本
npm run build
```

### 4.4 环境变量说明 (`.env`)

```ini
# ── LLM 文本模型配置 ──
GEMINI_API_KEY="your-api-key"          # 文本 LLM 的 API Key
GEMINI_BASE_URL="https://..."          # API 地址（兼容 OpenAI 协议）
GEMINI_MODEL="gemini-2.5-flash"        # 文本模型名

# ── 视觉模型配置（可选）──
VISION_API_KEY="your-vision-key"       # 视觉模型的 API Key
VISION_BASE_URL="https://api.openai.com/v1/"
VISION_MODEL="gpt-4o"                  # 推荐: gpt-4o / gpt-4o-mini

# ── MySQL 配置（可选，用于对话历史）──
MYSQL_HOST="localhost"
MYSQL_PORT="3306"
MYSQL_USER="root"
MYSQL_PASSWORD="your-password"
MYSQL_DATABASE="docmind"

# ── OCR 配置（可选）──
TESSERACT_PATH="D:/software/Tesseract-OCR/tesseract.exe"
OCR_LANGUAGES="chi_sim+eng"
```

### 4.5 快速上手三步走

```bash
# Step 1: 启动后端
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# Step 2: 启动前端
cd frontend && npm run dev

# Step 3: 打开浏览器访问
# http://localhost:5173
```

> ? **使用流程**: 打开页面 → 知识库页面上传 PDF → 点击"Add to KB" → 等待进度条到 100% → 切换到 Chat 页面 → 开始提问！

---

## ? 5. RAG 策略详解

### 5.1 核心理念：推理驱动，而非向量驱动

DocMind 采用 **Vectorless RAG（无向量检索增强生成）** 策略，完全抛弃了传统的 Embedding + Vector Search 范式。

| 对比维度 | 传统 RAG | DocMind (PageIndex 风格) |
|---------|---------|--------------------------|
| 文档切分 | 固定大小 chunk 切分 | ? 不做 chunk 切分，以物理页面为单位 |
| 索引方式 | 向量嵌入 → 向量数据库 | ? 层级树索引 (JSON) |
| 检索方式 | 余弦相似度 top-k | ? LLM Agent 自主导航树结构 |
| Embedding | 需要 embedding 模型 | ? 零 embedding 依赖 |
| 向量数据库 | ChromaDB / Milvus / Pinecone 等 | ? 纯文件系统 JSON |
| 检索精度 | 语义近似，可能漏/误 | ? 精准页面定位 |
| 幻觉风险 | 可能召回不相关 chunk | ? LLM 主动筛选相关页面 |

### 5.2 索引策略：层级树索引 (Hierarchical Tree Index)

#### 文档解析流水线

```
PDF 文件
  │
  ├─① 提取 PDF 书签 (Outline)           ← 零 LLM 依赖，PyMuPDF 直接读
  │   └─ 自动构建层级目录树
  │      1. Introduction
  │      1.1 Background
  │      1.2 Related Work
  │      2. Method
  │      2.1 Architecture
  │      2.2 Training
  │      3. Experiments
  │      ...
  │
  ├─② 计算每层 page_from / page_to        ← 递归精确计算
  │   └─ 知道每个章节覆盖哪些物理页面
  │
  ├─③ 按页提取文本内容                     ← PyMuPDF per-page text
  │   └─ 逐页存储到树索引目录
  │
  ├─④ 渲染 PDF 页面为图片                 ← PyMuPDF render (2x DPI)
  │   └─ 供 Vision RAG 使用
  │
  └─⑤ 生成章节摘要 (LLM)                  ← 可选，为每个节点生成一句话摘要
      └─ 存入树的 summary 字段
```

#### 树索引存储结构

```json
{
  "doc_id": "abc123",
  "doc_name": "RMSNorm.pdf",
  "doc_description": "A paper about Root Mean Square Layer Normalization...",
  "type": "pdf",
  "page_count": 8,
  "structure": [
    {
      "title": "Introduction",
      "structure": "1",
      "level": 1,
      "start_index": 1,
      "end_index": 2,
      "summary": "Introduces the problem of training instability in deep networks...",
      "nodes": [
        {
          "title": "Background",
          "structure": "1.1",
          "level": 2,
          "start_index": 1,
          "end_index": 1,
          "summary": "Reviews LayerNorm and BatchNorm foundations...",
          "nodes": []
        }
      ]
    },
    {
      "title": "Method",
      "structure": "2",
      "level": 1,
      "start_index": 3,
      "end_index": 5,
      "summary": "Proposes RMSNorm as a simpler alternative...",
      "nodes": [...]
    }
  ],
  "pages": [
    {"page": 1, "content": "..."},
    {"page": 2, "content": "..."}
  ]
}
```

#### 索引文件布局

```
tree_index/
├── _meta.json                    # 全局索引元数据
├── abc123.json                   # 文档树结构 (JSON)
├── abc123_pages/                 # 逐页文本内容
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
└── abc123_images/               # 逐页渲染图片
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

### 5.3 检索策略：Agentic Tool Calling

DocMind 的检索不是简单的"语义搜索 top-k"，而是一个 **LLM Agent 自主决策的多轮工具调用过程**：

#### Agent 可用工具

| 工具 | 功能 | 参数 |
|------|------|------|
| `get_document` | 获取文档元数据 | `doc_id` |
| `get_document_structure` | 获取完整层级目录树 | `doc_id` |
| `get_page_content` | 获取指定页面文本 | `doc_id`, `pages` |

#### 检索流程

```
用户提问："对比 RMSNorm、LayerNorm 和 BatchNorm 的区别"

  ┌─────────────────────────────────────────────┐
  │ Step 1: Agent 调用 get_document()           │
  │   对每个文档获取：名称、描述、页数           │
  │   输出：RMSNorm.pdf (8p), LayerNorm.pdf (12p)│
  │        BatchNorm.pdf (10p)                  │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────┐
  │ Step 2: Agent 调用 get_document_structure() │
  │   对每个文档获取完整的层级树（只读骨架）     │
  │   RMSNorm: 1. Intro → 2. Method → 3. Exp... │
  │   LayerNorm: 1. Intro → 2. Approach → ...   │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────┐
  │ Step 3: Agent 自主决策哪些页面相关          │
  │   根据树结构，LLM 推理：                     │
  │   - RMSNorm 的 Method 章节 (p3-5) 含公式    │
  │   - LayerNorm 的 Approach 章节 (p2-6)       │
  │   - BatchNorm 的 Algorithm 章节 (p4-7)      │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────┐
  │ Step 4: Agent 调用 get_page_content()       │
  │   精准获取相关页面内容                       │
  │   get_page_content("rms_id", "3-5")         │
  │   get_page_content("ln_id", "2-6")          │
  │   get_page_content("bn_id", "4-7")          │
  └──────────────────┬──────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────┐
  │ Step 5: 综合所有信息，生成最终答案          │
  │   直接回答用户，公式用 LaTeX 渲染           │
  └─────────────────────────────────────────────┘
```

#### 容错降级机制

当 LLM 不可用时（API 余额不足、网络故障等），系统自动降级为 **关键词匹配检索**：

- 对查询和所有页面内容做 TF 关键词匹配
- 按相关度排序返回 top-k 页面
- 用检索到的内容构造 prompt 直接回答

---

## ? 6. 知识库隔离机制

DocMind 采用 **文档级隔离（Document-Level Isolation）**，确保不同文档之间的知识检索互不干扰。

### 6.1 隔离原理

```
文档 A (RMSNorm.pdf)              文档 B (LayerNorm.pdf)
┌─────────────────────┐          ┌─────────────────────┐
│ doc_id: "abc123"    │          │ doc_id: "def456"    │
│ structure: [...]    │          │ structure: [...]    │
│ pages/: 1.txt-8.txt │          │ pages/: 1.txt-12.txt│
│ images/: 1.jpg-8.jpg│          │ images/: 1-12.jpg   │
└─────────────────────┘          └─────────────────────┘
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
         ┌─────────────────────┐
         │  tree_index/_meta   │
         │  {                  │
         │    "abc123": {...}, │  ← 元数据全局可见
         │    "def456": {...}  │
         │  }                  │
         └─────────────────────┘
```

- **存储隔离**：每个文档以独立 `doc_id` 为命名空间，拥有独立的 JSON 文件 (`{doc_id}.json`)、独立页面文本目录 (`{doc_id}_pages/`) 和独立图片目录 (`{doc_id}_images/`)
- **检索隔离**：Agent 只在当前对话指定的文档集合中检索（`doc_ids` 参数），不会跨文档混淆内容
- **元数据共享**：`_meta.json` 全局索引仅存储轻量元数据（文档名、描述、页数），不含正文，用于 Agent 初始化时的文档发现
- **来源追溯**：每条回答都会标注信息来源（doc_name + pages），用户可以清楚地知道答案出自哪篇文档的哪几页

### 6.2 隔离示例

```
场景：上传 3 篇文档（RMSNorm、LayerNorm、BatchNorm）

用户提问："RMSNorm 和 LayerNorm 的区别是什么？"

Agent 行为：
  1. get_document("rms_id") → 获取 RMSNorm 元数据
  2. get_document("ln_id")  → 获取 LayerNorm 元数据
  3. 【不会调用 get_document("bn_id")】← LLM 自主判断 BatchNorm 不相关
  4. 只翻 RMSNorm 和 LayerNorm 的相关页面
  5. 回答中不会混入 BatchNorm 的内容
```

---

## ? 7. 检索策略、索引策略、解决了怎样的问题

### 7.1 传统 RAG 的行业痛点

| 痛点 | 描述 | 传统方案的问题 |
|------|------|--------------|
| **Chunk 切分困境** | 固定大小切割会打断语义连续性 | 公式被切开、表格跨 chunk 丢失上下文 |
| **语义漂移** | embedding 模型对专业术语理解偏差 | "RMSNorm" 和 "LayerNorm" 向量距离很近但完全不同 |
| **检索精确度不足** | top-k 近似检索返回不相关或冗余内容 | LLM 被喂入大量无关 context，回答质量下降 |
| **幻觉放大** | 错误召回的 chunk 被 LLM 当作事实 | 尤其在学术文档中，"看起来相关" ≠ "实际上相关" |
| **基础设施复杂** | 需要向量数据库、embedding 服务 | 部署成本高、维护麻烦 |
| **成本高昂** | embedding 调用 + 向量存储 + 向量检索 | 每篇文档都产生额外的 API 和存储开销 |

### 7.2 DocMind 的解决方案

#### 方案一：不做 Chunk，直接以物理页面为检索单元

**解决的问题**：Chunk 切分导致的语义断裂、公式/表格被截断

```
传统 RAG:
  Page 1: "RMSNorm computes..."  → chunk_1
  Page 2: "μ = 1/n Σ xi"         → chunk_2  ← 公式被切断
  Page 2: "σ = sqrt(1/n Σ ...)"  → chunk_3  ← 上下文丢失

DocMind:
  Page 1: 完整保留 → page 1
  Page 2: 完整保留 → page 2   ← 公式完整、上下文完整
```

#### 方案二：层级树索引替代向量索引

**解决的问题**：语义漂移、近似检索不精确

```
向量检索流程:
  "RMSNorm formula" → embedding → [0.23, -0.45, ...] → cosine → 3个"相关"chunk
  问题：可能召回 LayerNorm 的 chunk（因为语义接近）

Agentic 检索流程:
  "RMSNorm formula" → LLM 看树结构 → 定位到 "Method → RMSNorm Definition" → 翻 p3-5
  优势：精准定位，不会混淆文档
```

#### 方案三：LLM Agent 自主导航替代固定检索逻辑

**解决的问题**：检索策略不够灵活，无法适应复杂的多文档交叉对比问题

```
传统检索:
  query → embedding → top-5 chunks → prompt → answer
  （固定流式，无法调整）

Agentic 检索:
  query → LLM 规划 → 多轮工具调用 → 动态调整检索策略 → answer
  （根据文档结构、问题复杂度自适应调整）
```

#### 方案四：Vision RAG 处理公式和表格

**解决的问题**：OCR / 文本提取对复杂公式和表格的识别差

```
文本提取:   "u = 1n i=1n xi"     ← 公式完全变形
Vision模型: 直接看页面图片 → "$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$$"
            完美保留排版
```

### 7.3 技术架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户界面 (React)                         │
│   KnowledgeBase │ Chat │ Search │ Memory │ OCR                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP / SSE Streaming
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI 后端 (Python)                        │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │documents │ │  chat    │ │ search   │ │  memory  │  OCR     │
│  │   API    │ │   API    │ │   API    │ │   API    │  API     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │            │            │            │                  │
│  ┌────▼─────┐ ┌───▼──────────▼──────────┐ ┌─▼──────────┐     │
│  │doc_handler│ │  Agentic Retriever     │ │Memory Svc  │     │
│  │ + parser  │ │  (LLM Tool Calls)      │ │(4-type)    │     │
│  └────┬─────┘ └───┬─────────┬──────────┘ └─┬──────────┘     │
│       │           │         │              │                  │
│  ┌────▼─────┐    │    ┌────▼─────┐    ┌───▼──────┐          │
│  │Tree Index│?───┘    │  Vision  │    │  MySQL   │          │
│  │  (JSON)  │         │  Model   │    │  (Chat)  │          │
│  └──────────┘         └──────────┘    └──────────┘          │
│                                                                  │
│  无向量数据库 · 无 Embedding 模型 · 纯文件系统存储              │
└──────────────────────────────────────────────────────────────────┘
```

### 7.4 关键设计决策

1. **零 LLM 书签提取**：PDF 解析优先使用 PyMuPDF 读取文档内嵌的书签/大纲（outline），完全不依赖 LLM。只有当 PDF 没有书签时，才 fallback 到 LLM 驱动的内容结构分析。这样做是为了确保即使 LLM API 不可用，系统核心的树索引构建仍能正常工作。

2. **层级完整性**：书签提取结果保留原始嵌套层级（4 级甚至更深），递归计算每个节点的 `page_from` 和 `page_to`，确保 Agent 能精确定位到任意深度的子章节。

3. **Keyword Fallback**：当 LLM Agent 的 tool calling 不可用时（余额不足、网络故障），自动降级为 TF 关键词匹配，确保检索功能不中断。

4. **流式 + 可中断**：使用 `AsyncOpenAI` + `async for` 实现真正的 token-by-token 流式输出；通过 `active_streams` 字典 + `asyncio.CancelledError` 实现用户点击 Stop 后立即终止生成。

---

## ? License

MIT License

---

## ? 致谢

DocMind 的设计理念深受 [PageIndex](https://github.com/nick-cao/PageIndex) 项目的启发，感谢其开创性的 Vectorless RAG 思想。
