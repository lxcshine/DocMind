# DocMind

> 一个能读文档、会翻页、能记住你偏好的研究型 AI Agent

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178c6.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. 项目简介

**DocMind 是一个研究型 AI Agent**，不是一个 RAG 框架。

大多数"文档问答系统"的设计思路是：把文档切成碎片 → 转成向量 → 塞进数据库 → 用户提问时搜几个碎片 → 喂给 LLM。这个过程叫 RAG，但它本质上是一个**检索器 + 生成器**的机械拼接，LLM 在整个链路中只是最后一步的"润色工具"。

DocMind 的思路完全相反：**LLM 是 Agent 的大脑，从头到尾主导整个过程。** 它像人类研究者一样工作——拿到一篇论文，先翻目录了解结构，找到感兴趣的章节，翻开对应页面阅读，然后综合所有信息给出答案。整个过程没有向量、没有 embedding、没有 chunk 切分。

```
传统 RAG 系统：                          DocMind Agent：
                                         
文档 → 切 chunk → 向量化 → 存库          文档 → 读目录 → 建树索引
  ↓                                        ↓
用户提问 → 向量化 → 搜 top-k             用户提问 → Agent 自主规划
  ↓                                        ↓
拼接 chunk + 提问 → LLM → 回答           Agent 翻目录 → 定位章节 → 翻页面
                                           ↓
                                         Agent 综合信息 → 回答
                                         
（LLM 只是最后的"润色器"）                （LLM 全程主导，像真人一样翻阅文档）
```

> 项目原名 ResearchFlow，现已更名为 DocMind。设计理念深受 [PageIndex](https://github.com/nick-cao/PageIndex) 的 Vectorless RAG 思想启发。

---

## 2. Agent 能力清单

DocMind Agent 具备以下核心能力：

| 能力 | 描述 | 实现方式 |
|------|------|---------|
| **📖 阅读文档** | 逐页读取 PDF / Markdown / TXT 文档内容 | Agent 工具调用 `get_page_content` |
| **🗂️ 浏览目录** | 查看文档的层级树骨架，快速定位章节 | Agent 工具调用 `get_document_structure` |
| **🔍 自主检索** | 根据问题自主规划翻阅哪些文档的哪些页面 | LLM 多轮 tool calling |
| **👁️ 看图理解** | 对包含公式/表格的页面，调用视觉模型直接"看" | Vision Model 按需调用 |
| **🧠 长期记忆** | 记住用户的研究兴趣、偏好格式、历史提问 | 4 类记忆体系 + 关键词索引 |
| **💬 流式对话** | 逐字输出回答，支持随时打断 | AsyncOpenAI SSE 流式 |
| **📝 OCR 识别** | 对扫描版 PDF / 图片 / PPT 做文字识别 + 智能纠错 | Tesseract + LLM 后处理 |
| **🌐 网页抓取** | 抓取网页内容并转为 Markdown | 多引擎 Firecrawl 风格 |
| **💾 会话持久化** | 对话历史存 MySQL，支持多会话管理 | MySQL + 会话级隔离 |

### 双模型调度

Agent 内部运行两个模型，按需切换：

```
文本模型（GLM）  ← 默认模式，处理纯文本问答、文档解析、对话生成
                    成本低，响应快

视觉模型（VLM）  ← 仅在检测到公式/表格时激活
                    直接看 PDF 页面图片，完美保留公式排版
                    正则自动检测，无需人工干预
```

---

## 3. 项目结构

```
DocMind/
├── backend/                          # Python 后端 (FastAPI)
│   ├── main.py                       # 应用入口，Agent 初始化
│   ├── requirements.txt              # Python 依赖
│   ├── .env.example                  # 环境变量模板
│   ├── config/
│   │   └── settings.py               # 全局配置（模型、路径、阈值）
│   ├── api/                          # Agent 对外接口层
│   │   ├── documents.py              # 文档上传/处理/删除/列表
│   │   ├── chat.py                   # 核心对话接口 (/chat, /stream)
│   │   ├── search.py                 # 智能检索接口
│   │   ├── memory.py                 # 记忆管理接口
│   │   └── ocr.py                    # OCR 识别接口
│   ├── core/                         # Agent 大脑核心
│   │   ├── parser.py                 # 文档解析器（零 LLM 书签提取）
│   │   ├── doc_handler.py            # 文档处理流水线（上传→解析→建树→存储）
│   │   ├── agentic_retrieve.py       # Agent 检索器（工具调用 + 规划）
│   │   ├── tree_index.py             # 层级树索引（JSON 文件系统）
│   │   ├── meta_store.py             # 文档元数据管理
│   │   ├── progress.py               # 处理进度追踪
│   │   ├── chat_history.py           # MySQL 对话历史
│   │   ├── memory.py                 # 长期记忆（4 类 + 关键词索引）
│   │   ├── ocr_handler.py            # 智能 OCR 处理器
│   │   └── web_scraper.py            # 网页抓取引擎
│   ├── tree_index/                   # 树索引持久化目录
│   ├── uploads/                      # 上传文件存储
│   └── tests/                        # 测试文件
├── frontend/                         # React 前端 (Vite + TypeScript + Ant Design)
│   ├── src/
│   │   ├── App.tsx                   # 路由配置
│   │   ├── layouts/
│   │   │   └── MainLayout.tsx        # 主布局
│   │   └── pages/
│   │       ├── KnowledgeBase/        # 知识库管理页面
│   │       ├── Chat/                 # Agent 对话页面
│   │       ├── Search/               # 智能检索页面
│   │       ├── Memory/               # 记忆管理页面
│   │       └── OCR/                  # OCR 识别页面
│   └── package.json
└── README.md
```

---

## 4. 安装与使用

### 4.1 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.10+ | 后端运行环境 |
| Node.js | 18+ | 前端运行环境 |
| MySQL | 5.7+ | 可选，用于对话历史持久化 |
| Tesseract OCR | 5.x | 可选，用于 OCR 功能 |

### 4.2 后端安装

```bash
cd backend

# 创建虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 启动
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4.3 前端安装

```bash
cd frontend
npm install
npm run dev        # 开发模式
npm run build      # 生产构建
```

### 4.4 环境变量

```ini
# ── 文本模型（必需）──
GEMINI_API_KEY="your-api-key"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL="gemini-2.5-flash"

# ── 视觉模型（可选，用于公式/表格渲染）──
VISION_API_KEY="your-vision-key"
VISION_BASE_URL="https://api.openai.com/v1/"
VISION_MODEL="gpt-4o"

# ── MySQL（可选，对话历史）──
MYSQL_HOST="localhost"
MYSQL_PORT="3306"
MYSQL_USER="root"
MYSQL_PASSWORD="your-password"
MYSQL_DATABASE="docmind"

# ── OCR（可选）──
TESSERACT_PATH="D:/software/Tesseract-OCR/tesseract.exe"
OCR_LANGUAGES="chi_sim+eng"
```

### 4.5 快速上手

```bash
# 终端 1：启动后端
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端
cd frontend && npm run dev

# 浏览器打开 http://localhost:5173
# 流程：上传 PDF → 点击 "Add to KB" → 等待完成 → 切换到 Chat 开始对话
```

---

## 5. Agent 如何阅读文档——"RAG 策略"的真相

> **我们不是为了用 RAG 而用 RAG。** Agent 需要阅读文档，RAG 只是实现"阅读"这个能力的一种手段。而传统的 chunk + embedding + vector search 这种手段，在学术文档场景下问题太多。

### 5.1 为什么不做 Chunk 切分

一个学术论文的典型页面：

```
3.2 Root Mean Square Layer Normalization

The root mean square statistic can be computed as:

    RMS(x) = sqrt(1/n * Σ_{i=1}^{n} x_i^2)      (1)

Unlike LayerNorm which centers and scales the inputs...

The gradient of RMSNorm with respect to input x_i is:

    ∂L/∂x_i = (g_i / RMS(x)) * (∂L/∂ŷ_i - (ŷ_i / n) * Σ_{j=1}^{n} ŷ_j * ∂L/∂ŷ_j)   (2)
```

如果按 512 token 切 chunk，公式 (1) 和 (2) 可能被切到不同 chunk 里，导数推导的上下文完全断裂。更糟的是，embedding 模型根本看不懂这些公式——`\frac{\partial L}{\partial x_i}` 在向量空间里可能和 `\frac{\partial L}{\partial w}` 距离一致。

**DocMind 的答案：不做 chunk。** 以物理页面为最小检索单元。一个页面就是一个完整的上下文单元，不管上面是文字、公式还是表格，都原封不动地保留。

### 5.2 索引策略：给 Agent 画一张地图

Agent 阅读文档前，需要的不是一堆向量，而是一张**地图**——知道文档有哪些章节，每个章节在哪些页。

```
Agent 看到的地图（树索引骨架）：

RMSNorm.pdf (8 pages)
├── 1. Introduction               [p1-2]
│   ├── 1.1 Background            [p1]
│   └── 1.2 Related Work          [p2]
├── 2. Method                     [p3-5]
│   ├── 2.1 RMSNorm Definition    [p3]     ← 核心公式在这里
│   ├── 2.2 Gradient Analysis     [p4]     ← 梯度推导在这里
│   └── 2.3 Implementation        [p5]
└── 3. Experiments                [p6-8]
    ├── 3.1 Setup                 [p6]
    └── 3.2 Results               [p7-8]
```

这张地图的构建过程：

```
PDF 文件
  │
  ├─① 提取 PDF 书签 (Outline)          ← PyMuPDF 直接读，零 LLM 调用
  │   └─ 自动构建层级目录树
  │      保留原始嵌套层级（可达 4 级甚至更深）
  │
  ├─② 递归计算每层的 page_from / page_to
  │   └─ 知道每个章节精确覆盖哪些物理页面
  │
  ├─③ 按页提取文本                       ← PyMuPDF per-page text
  │   └─ 逐页存储为独立 .txt 文件
  │
  ├─④ 渲染 PDF 页面为图片                ← 2x DPI，供视觉模型使用
  │
  └─⑤ 生成章节摘要 (LLM, 可选)           ← 为每个节点生成一句话摘要
      └─ 存入树节点的 summary 字段
```

索引存储为纯 JSON 文件，无数据库依赖：

```
tree_index/
├── _meta.json                    # 全局元数据索引
├── abc123.json                   # 文档树结构
├── abc123_pages/                 # 逐页文本
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
└── abc123_images/               # 逐页渲染图片
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

### 5.3 检索策略：Agent 自己翻书，不需要"搜索引擎"

这是 DocMind 与传统 RAG 最本质的区别。传统 RAG 的检索是**一个搜索引擎**——用户输入 query，系统返回 top-k 个"最相似"的 chunk。DocMind 的检索是**Agent 自己在翻书**——Agent 拿到地图后，自己决定翻哪些页。

Agent 拥有 3 个工具：

| 工具 | 功能 | 类比 |
|------|------|------|
| `get_document(doc_id)` | 获取文档元数据（名称、描述、页数） | 看书名和摘要 |
| `get_document_structure(doc_id)` | 获取完整层级目录树 | 翻到目录页 |
| `get_page_content(doc_id, pages)` | 获取指定页面的完整文本 | 翻到具体某一页 |

一次典型的检索对话：

```
用户："对比 RMSNorm、LayerNorm 和 BatchNorm 的核心公式差异"

Agent 的思考过程：

  Round 1: get_document("rms_id"), get_document("ln_id"), get_document("bn_id")
  → 了解三篇文档的基本信息

  Round 2: get_document_structure("rms_id"), get_document_structure("ln_id"),
           get_document_structure("bn_id")
  → 获取三篇文档的目录结构，Agent 发现：
    - RMSNorm: 2.1 Definition (p3) 含公式
    - LayerNorm: 2. Method (p2-4) 含公式
    - BatchNorm: 3. Algorithm (p4-5) 含公式

  Round 3: get_page_content("rms_id", "3"), get_page_content("ln_id", "2-4"),
           get_page_content("bn_id", "4-5")
  → 精准翻阅相关页面，不碰其他无关内容

  Round 4: 综合所有信息，生成对比回答
```

**关键点**：Agent 不是被动地接收搜索结果，而是**主动规划**要翻哪些页。它能看到目录结构，理解文档的章节组织，像一个真正的研究者那样做出判断。

### 5.4 为什么这比向量检索更好

| 问题场景 | 向量检索的表现 | Agent 的表现 |
|---------|-------------|------------|
| 用户问"RMSNorm 公式" | 可能召回 LayerNorm 的 chunk（语义接近，向量距离近） | Agent 看目录，精准定位到 RMSNorm 的 Definition 章节 |
| 需要对比 3 篇文档 | 3 篇文档的 chunk 混在一起 top-k，无法区分来源 | Agent 分别翻 3 篇文档的对应章节，来源清晰 |
| 文档包含大量公式 | embedding 看不懂公式，向量质量差 | Agent 翻到公式所在页面，调用视觉模型直接"看" |
| 需要跨章节综合 | top-k 可能遗漏关键章节 | Agent 根据目录结构自主判断需要翻哪些章节 |

### 5.5 容错：当 LLM 不可用时

当 LLM API 余额不足或网络故障时，Agent 自动降级为**关键词匹配模式**：

- 对查询词和所有页面内容做 TF 关键词匹配
- 按相关度排序，返回 top-k 页面
- 用检索到的内容构造 prompt 直接回答

这确保了即使 LLM 不可用，核心的文档检索功能仍然可以工作。

---

## 6. 知识库隔离

DocMind 采用**文档级隔离**。每个文档是一个独立的命名空间，Agent 在翻阅时不会混淆文档边界。

### 6.1 隔离机制

```
文档 A: RMSNorm.pdf               文档 B: LayerNorm.pdf
┌──────────────────────┐         ┌──────────────────────┐
│ doc_id: "abc123"     │         │ doc_id: "def456"     │
│ abc123.json          │         │ def456.json          │
│ abc123_pages/        │         │ def456_pages/        │
│ abc123_images/       │         │ def456_images/       │
└──────────────────────┘         └──────────────────────┘
         │                                  │
         └────────────┬────────────────────┘
                      ▼
            ┌──────────────────┐
            │  _meta.json      │
            │  {               │
            │   "abc123": {...}│  ← 只存元数据，不含正文
            │   "def456": {...}│
            │  }               │
            └──────────────────┘
```

三层隔离保证：

1. **存储隔离**：每个 `doc_id` 拥有独立的 JSON 文件、页面文本目录、图片目录
2. **检索隔离**：Agent 的每次工具调用都携带明确的 `doc_id`，不会跨文档混淆
3. **来源追溯**：每条回答标注 `doc_name + pages`，用户清楚知道信息来自哪篇文档的哪几页

### 6.2 实际效果

```
场景：知识库中有 RMSNorm、LayerNorm、BatchNorm 三篇文档

用户："RMSNorm 和 LayerNorm 的区别是什么？"

Agent 行为：
  1. get_document("rms_id")  → 获取 RMSNorm 信息
  2. get_document("ln_id")   → 获取 LayerNorm 信息
  3. 自主判断 BatchNorm 与当前问题不相关，不翻阅
  4. 回答中不会混入任何 BatchNorm 的内容
```

---

## 7. 检索策略、索引策略、解决什么问题

> 这一节从"问题驱动"的角度解释：DocMind 的每个设计决策，对应解决了传统方案中的哪个具体痛点。

### 7.1 痛点一：Chunk 切分破坏语义完整性

**问题**：学术文档中，一个公式的推导可能跨越 3 页，一张表格可能有 20 行。固定大小切分会把它们切成碎片，LLM 看到的是"断章取义"的上下文。

**解决方案**：以物理页面为最小检索单元。一个页面就是一个完整的上下文窗口，不做切分。

```
传统方案：                    DocMind：
Page 1: "...RMSNorm..."       Page 1: 完整保留
Page 2: "RMS(x) = ..."        Page 2: 完整保留  ← 公式完整
Page 2: "∂L/∂x_i = ..."       Page 3: 完整保留  ← 推导完整
Page 3: "...conclusion"

切分后：
chunk_1: Page 1 前半段
chunk_2: Page 1 后半段 + Page 2 前半段  ← 公式被切断
chunk_3: Page 2 后半段 + Page 3 前半段  ← 推导被切断
```

### 7.2 痛点二：Embedding 看不懂专业内容

**问题**：通用 embedding 模型对学术术语、数学公式、领域专业名词的语义理解很差。`RMSNorm` 和 `LayerNorm` 在向量空间中可能非常接近，但它们是完全不同的技术。

**解决方案**：不做 embedding。Agent 通过阅读文档的**目录结构**（而非向量相似度）来判断相关性。目录是人写的，天然反映了文档的逻辑组织。

```
向量检索：                     Agent 检索：
"RMSNorm formula"              "RMSNorm formula"
  → embedding                  → 看 RMSNorm 的目录
  → [0.23, -0.45, 0.67, ...]  → 定位到 "2.1 RMSNorm Definition"
  → cosine similarity          → 翻到 p3
  → 返回 3 个"相关"chunk       → 精准获取公式内容
  （可能包含 LayerNorm chunk）  （不可能混淆文档）
```

### 7.3 痛点三：固定检索逻辑无法应对复杂问题

**问题**：传统 RAG 的检索是固定的——query → embedding → top-k → 拼接 → LLM。这个流程无法适应"对比 3 篇文档的第 2 章和第 4 章"这种需要多轮、多文档、跨章节导航的复杂问题。

**解决方案**：Agent 自主规划检索路径。LLM 看到问题后，先理解意图，再决定翻哪些文档、哪些章节、哪些页面。这是一个**多轮决策过程**，而不是一次性的搜索。

```
传统检索：                      Agent 检索：
query → embedding → top-5       query → 理解意图
  → 拼接 → LLM → 回答            → 规划：需要翻 A 的 p3-5, B 的 p2-4, C 的 p4-7
（固定流程，无法调整）            → 执行：翻页 → 阅读 → 综合 → 回答
                                （自适应，根据问题复杂度调整）
```

### 7.4 痛点四：OCR 文本提取对公式/表格的识别差

**问题**：从 PDF 提取文本时，复杂公式会变成乱码（`∑_{i=1}^{n}` → `i=1n`），表格的列对齐完全丢失。

**解决方案**：双模型架构。Agent 检测到内容包含公式或表格时，自动切换到视觉模型，直接"看"页面图片。视觉模型能完美保留公式排版和表格结构。

```
文本提取：                      视觉模型：
"u = 1n i=1n xi"              直接看页面图片
（公式完全变形）                → "$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$$"
                                （完美保留 LaTeX 排版）

|A|B|C|                       直接看页面图片
|1|2|3|                        → 识别为 Markdown 表格
（表格列对齐丢失）               （完整保留表格结构）
```

### 7.5 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                      用户界面 (React)                         │
│   KnowledgeBase │ Chat │ Search │ Memory │ OCR               │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP / SSE Streaming
┌───────────────────────────▼──────────────────────────────────┐
│                  FastAPI 后端 — Agent 运行环境                 │
│                                                               │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐          │
│  │ 文档管理 │ │ 对话接口  │ │ 检索接口  │ │ 记忆接口 │  OCR    │
│  └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬────┘          │
│       │           │            │            │                 │
│  ┌────▼────┐ ┌───▼────────────▼────────┐ ┌──▼──────────┐    │
│  │ 文档解析 │ │   Agentic Retriever    │ │ Memory Svc  │    │
│  │ + 建树  │ │   (LLM 工具调用 + 规划)  │ │ (4 类记忆)  │    │
│  └────┬────┘ └───┬──────────┬─────────┘ └──┬──────────┘    │
│       │          │          │              │                 │
│  ┌────▼────┐    │     ┌────▼─────┐   ┌────▼─────┐          │
│  │Tree Index│◄──┘     │  Vision  │   │  MySQL   │          │
│  │  (JSON)  │         │  Model   │   │ (会话)   │          │
│  └─────────┘          └──────────┘   └──────────┘          │
│                                                               │
│  零向量数据库 · 零 Embedding · 零 Chunk 切分 · 纯文件系统      │
└───────────────────────────────────────────────────────────────┘
```

### 7.6 关键设计决策

1. **零 LLM 书签提取**：PDF 解析优先用 PyMuPDF 读取文档内嵌书签，完全不依赖 LLM。只有 PDF 无书签时，才 fallback 到 LLM 结构分析。这确保即使 LLM API 不可用，核心的树索引构建仍能工作。

2. **层级完整性**：书签保留原始嵌套层级（可达 4 级或更深），递归计算每个节点的 `page_from` / `page_to`，Agent 能精确定位到任意深度的子章节。

3. **Keyword Fallback**：LLM 不可用时，自动降级为 TF 关键词匹配，检索功能不中断。

4. **流式 + 可中断**：`AsyncOpenAI` + `async for` 实现真正的 token-by-token 流式输出；`active_streams` + `asyncio.CancelledError` 实现用户 Stop 后立即终止。

---

## 8. 上下文记忆——Agent 如何记住你

> 一个真正的 Agent 不能每次对话都从零开始。它需要记住用户是谁、研究什么方向、偏好什么格式、之前问过什么问题。

### 8.1 双层记忆架构

DocMind 的上下文记忆由两层组成：

```
┌─────────────────────────────────────────────┐
│              第一层：会话记忆               │
│  ┌───────────────────────────────────────┐  │
│  │ MySQL chat_messages 表                │  │
│  │ session_id → [user, assistant, ...]   │  │
│  │ 每次对话时取最近 20 轮作为上下文       │  │
│  │ 作用：让 Agent 知道"刚才在聊什么"     │  │
│  └───────────────────────────────────────┘  │
│                                             │
│              第二层：长期记忆               │
│  ┌───────────────────────────────────────┐  │
│  │ memory_db.json + MemoryIndex          │  │
│  │ 4 类记忆 + 关键词索引                 │  │
│  │ 作用：让 Agent 知道"这个用户是谁"     │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 8.2 长期记忆的四种类型

借鉴 RAGFlow 的记忆模块设计，DocMind 将长期记忆分为四类：

| 记忆类型 | 存储内容 | 示例 |
|---------|---------|------|
| **Raw（原始）** | 原始对话记录 | 完整保留每一轮 user/assistant 消息 |
| **Semantic（语义）** | 用户提到的领域知识、研究兴趣 | "用户正在研究 Transformer 架构中的归一化技术" |
| **Episodic（情景）** | 用户的具体提问、探索轨迹 | "用户曾对比过 RMSNorm 和 LayerNorm 的梯度特性" |
| **Procedural（程序性）** | 用户的偏好格式、交互习惯 | "用户偏好 LaTeX 公式用 $$ 块级渲染，答案用 Markdown 表格" |

### 8.3 记忆的生命周期

```
每次对话结束
  │
  ├─① 保存原始对话到 MySQL（会话记忆）
  │
  ├─② LLM 自动从对话中提取记忆
  │   ┌──────────────────────────────────────┐
  │   │ 输入：User 消息 + Assistant 回答      │
  │   │ 输出：结构化 JSON                     │
  │   │ {                                    │
  │   │   "semantic": [                      │
  │   │     {"content": "...", "keywords":   │
  │   │      ["RMSNorm", "normalization"]}   │
  │   │   ],                                 │
  │   │   "episodic": [...],                 │
  │   │   "procedural": [...]                │
  │   │ }                                    │
  │   └──────────────────────────────────────┘
  │
  ├─③ 关键词向量化 → 存入 MemoryIndex
  │   └─ 用于后续快速检索
  │
  └─④ FIFO 遗忘策略
      └─ 超过 max_entries 时，自动删除最早的记忆
```

### 8.4 记忆如何影响对话

每次用户提问时，Agent 会同时做两件事：

1. **检索相关记忆**：用用户提问的关键词在 MemoryIndex 中搜索 top-5 最相关的长期记忆
2. **注入 System Prompt**：将检索到的记忆内容注入到 System Prompt 中

```
System Prompt 示例：

"You are a helpful research assistant...

**User Profile & Past Context (from Memory):**
[Semantic Knowledge] 用户正在研究深度学习中的归一化技术
[Episodic Memory] 用户曾对比过 RMSNorm 和 LayerNorm 的梯度特性
[Procedural Memory] 用户偏好 LaTeX 公式用 $$ 渲染，答案用 Markdown 表格

Use this information to personalize your response."
```

### 8.5 为什么不是简单的"把历史对话全塞进去"

很多 RAG 系统简单地把最近 N 轮对话拼接到 prompt 里，这有两个问题：

1. **上下文窗口浪费**：无差别地塞入所有历史对话，大量 token 浪费在寒暄和无关内容上
2. **缺乏结构化理解**：LLM 需要自己从原始对话中推断用户偏好，效率低且不稳定

DocMind 的做法是**先提取、再检索**：

- 用 LLM 从对话中提取结构化的记忆（语义/情景/程序性）
- 存储时带上关键词索引
- 每次对话时，只检索与当前问题最相关的 top-5 条记忆注入 prompt

这样既节省了上下文窗口，又让 Agent 对用户有结构化的理解——不只是"你刚才说了什么"，而是"你是谁、研究什么、偏好什么"。

### 8.6 记忆管理界面

用户可以在前端 Memory 页面：

- 查看所有记忆条目，按类型筛选
- 启用/禁用特定记忆（不想让 Agent 记住的可以暂时关掉）
- 手动删除不再需要的记忆
- 查看每种记忆类型的统计

---

## 📄 License

MIT License

---

## 🙏 致谢

DocMind 的设计理念深受 [PageIndex](https://github.com/nick-cao/PageIndex) 项目 Vectorless RAG 思想的启发，记忆模块借鉴了 [RAGFlow](https://github.com/infiniflow/ragflow) 的多类型记忆设计。感谢这些项目的开创性工作。
