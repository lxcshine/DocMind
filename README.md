# DocMind

> 一个能读文档、会翻页、能记住你偏好的研究型 AI Agent

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 目录

1. [项目简介](#1-项目简介)
2. [核心能力](#2-核心能力)
3. [项目结构](#3-项目结构)
4. [安装与使用](#4-安装与使用)
5. [Agent 如何阅读文档](#5-agent-如何阅读文档)
6. [知识库隔离](#6-知识库隔离)
7. [设计决策：检索策略与索引策略](#7-设计决策检索策略与索引策略)
8. [上下文记忆](#8-上下文记忆)
9. [致谢](#9-致谢)

---

## 1. 项目简介

DocMind 是一个面向学术文档场景的研究型 AI Agent。它的核心命题不是"如何把文档塞进向量数据库然后搜出来"，而是**如何让 AI 像一个有经验的研究者那样阅读、理解和思考文档**。

现有的文档问答方案几乎全部建立在 RAG 范式之上：文档切片、embedding 向量化、近似最近邻搜索、top-k 拼接、喂给 LLM 生成回答。这条链路在 FAQ 客服、产品说明书等场景下工作良好，但在学术文档面前暴露了三个根本性缺陷：

**第一，语义单元的完整性被破坏。** 学术论文的一节推导可能横跨三页，一张对比表格可能包含二十行数据。固定长度的 chunk 切分把推导切成了互不连贯的片段，LLM 看到的是残篇，而非完整的论证过程。

**第二，embedding 模型理解不了专业内容。** 通用向量模型对数学公式、领域术语、算法伪代码的语义表征能力极其有限。`RMSNorm` 和 `LayerNorm` 在向量空间中可能近在咫尺，但它们讨论的是两种完全不同的归一化策略。向量距离在这里提供的不是相关性信号，而是噪音。

**第三，检索逻辑是固化的，无法适应复杂的信息需求。** 当用户问"对比这三篇论文第二章和第四章的实验设置差异"时，传统 RAG 只能做一次 top-k 搜索，它既不理解问题中的多文档跨章节意图，也不具备规划多轮检索的能力。

DocMind 的出发点是：**把 LLM 从最后一步的"润色工具"提升到全程主导的"思考中枢"。** Agent 拿到一篇论文后，先翻阅目录了解全貌，定位到感兴趣的章节，翻开具体页面阅读，最后综合所有信息形成回答。整个过程没有向量、没有 embedding、没有 chunk 切分。

```
传统 RAG（LLM 作为润色器）            DocMind（LLM 作为 Agent 大脑）

文档 → 切片 → 向量化 → 入库           文档 → 建目录树 → 逐页提取
  ↓                                    ↓
用户提问 → 向量化 → top-k 检索        用户提问 → Agent 理解意图
  ↓                                    ↓
拼接 chunk → LLM → 回答               Agent 翻目录 → 定位章节 → 翻页
                                        ↓
                                      Agent 综合 → 回答
```

项目原名 ResearchFlow，现已更名为 DocMind。设计理念受到 [PageIndex](https://github.com/nick-cao/PageIndex) Vectorless RAG 思想的深刻影响，记忆模块的设计参考了 [RAGFlow](https://github.com/infiniflow/ragflow) 的多类型记忆架构。

---

## 2. 核心能力

DocMind Agent 的功能围绕"阅读—理解—记忆"这条主线展开。每个能力不是一个孤立的功能点，而是服务于 Agent 完成一次高质量研究对话所需的完整链路。

### 文档阅读与导航

Agent 通过三个工具完成对文档的自主导航：

- **`get_document`** — 获取文档元数据（名称、页数、描述），相当于看书名和摘要。
- **`get_document_structure`** — 获取完整的层级目录树，每个节点标注了页码范围。Agent 据此判断文档的章节组织，定位目标内容所在的精确页面。
- **`get_page_content`** — 获取任意指定页面的完整内容。Agent 自主决定翻阅哪些页面、翻阅多少页，而不是被动接收搜索引擎返回的"最相似片段"。

三个工具的设计保证了 Agent 拥有和人类研究者同等的"翻书"自由度。

### 双模型调度

Agent 内部同时接入文本模型和视觉模型，根据内容的性质自动切换：

- **文本模型**负责常规的文本理解、文档解析和对话生成。成本低，响应快，覆盖 90% 的交互场景。
- **视觉模型**仅在检测到页面包含公式或表格时激活。它将 PDF 页面渲染为高分辨率图片，直接"看"页面内容，完全绕过了文本提取对数学符号和表格结构的破坏。

切换逻辑由正则表达式自动驱动，无需用户干预。Agent 在生成回答时会自动判断："这一页全是文字，走文本模型就够了"还是"这一页有求和符号和分式，必须让 VLM 看看"。

### 长期记忆

Agent 不会在每次对话后重置。它从对话中提取四类记忆并持久化：

- **Raw Memory** — 原始对话记录，忠实保留用户与 Agent 的每一轮交互。
- **Semantic Memory** — 用户的知识偏好和研究兴趣，如"关注归一化方法的理论推导"。
- **Episodic Memory** — 对话中的关键事件和决策过程，如"上次对比了 RMSNorm 和 LayerNorm"。
- **Procedural Memory** — 用户的交互偏好，如回答格式、详细程度、引用风格。

记忆以关键词索引的方式组织，检索时返回与当前对话最相关的 top-5 记忆条目，注入 System Prompt。Agent 因此能追踪用户的兴趣演变，避免重复提问。

### 其他能力

- **流式对话** — 基于 SSE（Server-Sent Events）的逐字输出，支持用户随时打断并切换话题。
- **OCR 识别** — 对扫描版 PDF、图片、PPT 截图做文字识别，Tesseract 引擎配合 LLM 后处理纠错。
- **网页抓取** — 将网页内容转换为 Markdown，作为知识库的补充信息源。
- **会话持久化** — 对话历史存入 MySQL，支持创建、切换和删除会话。

---

## 3. 项目结构

```
DocMind/
├── backend/                          # Python 后端
│   ├── main.py                       # 应用入口，Agent 初始化
│   ├── requirements.txt
│   ├── .env.example
│   ├── config/
│   │   └── settings.py               # 全局配置（模型、路径、阈值）
│   ├── api/                          # API 接口层
│   │   ├── chat.py                   # 核心对话接口（SSE 流式 + Vision 调度）
│   │   ├── documents.py              # 文档上传 / 处理 / 删除
│   │   ├── search.py                 # 智能检索接口
│   │   ├── memory.py                 # 记忆管理接口
│   │   └── ocr.py                    # OCR 识别接口
│   ├── core/                         # Agent 核心
│   │   ├── parser.py                 # 文档解析器（PDF 书签提取 + 页面范围计算）
│   │   ├── doc_handler.py            # 文档处理流水线（上传 → 解析 → 建树 → 存储）
│   │   ├── agentic_retrieve.py       # Agent 检索器（LLM 工具调用 + 关键词回退）
│   │   ├── tree_index.py             # 层级树索引（JSON 文件系统）
│   │   ├── meta_store.py             # 文档元数据
│   │   ├── progress.py               # 处理进度追踪
│   │   ├── chat_history.py           # MySQL 对话历史
│   │   ├── memory.py                 # 长期记忆（4 类 + 关键词索引 + FIFO）
│   │   ├── ocr_handler.py            # OCR 处理器
│   │   └── web_scraper.py            # 网页抓取
│   ├── tree_index/                   # 树索引持久化目录
│   ├── uploads/                      # 上传文件存储
│   └── tests/
├── frontend/                         # React 前端
│   ├── src/
│   │   ├── App.tsx
│   │   ├── layouts/
│   │   │   └── MainLayout.tsx
│   │   └── pages/
│   │       ├── KnowledgeBase/        # 知识库管理
│   │       ├── Chat/                 # Agent 对话
│   │       ├── Search/               # 智能检索
│   │       ├── Memory/               # 记忆管理
│   │       └── OCR/                  # OCR 识别
│   └── package.json
└── README.md
```

---

## 4. 安装与使用

### 4.1 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 后端运行环境 |
| Node.js | 18+ | 前端运行环境 |
| MySQL | 5.7+ | 可选，用于对话历史持久化 |
| Tesseract OCR | 5.x | 可选，用于 OCR 功能 |

### 4.2 后端

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 API Key 后保存
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4.3 前端

```bash
cd frontend
npm install
npm run dev          # 开发模式
npm run build        # 生产构建
```

### 4.4 环境变量

```ini
# 文本模型（必需）
GEMINI_API_KEY="your-api-key"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL="gemini-2.5-flash"

# 视觉模型（可选，用于公式和表格渲染）
VISION_API_KEY="your-vision-key"
VISION_BASE_URL="https://api.openai.com/v1/"
VISION_MODEL="gpt-4o"

# MySQL（可选，对话历史）
MYSQL_HOST="localhost"
MYSQL_PORT="3306"
MYSQL_USER="root"
MYSQL_PASSWORD="your-password"
MYSQL_DATABASE="docmind"

# OCR（可选）
TESSERACT_PATH="D:/software/Tesseract-OCR/tesseract.exe"
OCR_LANGUAGES="chi_sim+eng"
```

### 4.5 快速上手

启动后端和前端后，浏览器打开 `http://localhost:5173`。典型使用流程：在 KnowledgeBase 页面上传 PDF，点击 Add to KB 等待处理完成（状态变为 Completed），然后切换到 Chat 页面开始对话。

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
