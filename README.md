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
  <strong>RAG文档智能平台</strong>
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

![Homepage Screenshot](https://github.com/lxcshine/DocMind/blob/master/image/image_6.png)

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

## Docker 部署

```bash
# 确保 backend/.env 已配置
docker compose up -d

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f backend
```

**服务端口：**

| 服务 | 端口 | 说明 |
|------|------|------|
| Frontend | 80 | Nginx 静态资源 + API 代理 |
| Backend | 8000 | FastAPI + MCP Server |
| MySQL | 3306（内部） | 不暴露到宿主，仅后端通过内部网络访问 |

**数据持久化：**

| Volume | 用途 |
|--------|------|
| `mysql-data` | MySQL 数据 |
| `backend-data` | RAG 存储（向量、图谱） |
| `backend-output` | 文档解析输出 |
| `backend-uploads` | 上传文件 |
| `backend-logs` | 应用日志 |

---

### Kubernetes部署

DocMind 提供完整的 Kubernetes 部署方案，支持高可用、数据持久化、零停机滚动更新和自动扩缩容。

#### 前置条件

| 依赖 | 版本 | 说明 |
|------|------|------|
| Kubernetes | v1.24 | 集群已就绪 |
| kubectl | v1.24 | 已配置并连接到集群 |
| StorageClass | RWX 支持 | 后端 PVC 需要 ReadWriteMany（推荐 Longhorn / CephFS / NFS / AWS EFS） |
| Ingress Controller | nginx-ingress | 处理外部流量和 TLS 终结 |
| cert-manager | v1.12+ | 可选，自动 TLS 证书签发 |
| 容器镜像仓库 | — | 集群可访问的镜像仓库 |

#### 目录结构

```
k8s/
├── deploy.sh                          # 一键部署脚本
├── base/
│   ├── kustomization.yaml             # Kustomize 基础入口
│   ├── namespace.yaml                 # Namespace 定义
│   ├── configmap.yaml                 # 非敏感配置
│   ├── secret.yaml                    # 敏感配置 (API Keys, 密码)
│   ├── pvc.yaml                       # 持久化卷声明
│   ├── mysql.yaml                     # MySQL StatefulSet + Service
│   ├── backend.yaml                   # Backend Deployment + Service + PDB
│   ├── frontend.yaml                  # Frontend Deployment + Service + PDB
│   ├── ingress.yaml                   # Ingress (TLS + SSE 支持)
│   ├── hpa.yaml                       # HPA 自动扩缩容
│   └── networkpolicy.yaml             # 网络安全策略
└── overlays/
    └── production/
        └── kustomization.yaml         # 生产环境 Overlay
```

#### 快速部署

**1. 配置密钥**

编辑 `k8s/base/secret.yaml`，替换所有 `REPLACE_ME_` 占位符：

```yaml
stringData:
  GEMINI_API_KEY: "your-actual-api-key"
  VISION_API_KEY: "your-actual-api-key"
  EMBEDDING_API_KEY: "your-actual-api-key"
  MYSQL_ROOT_PASSWORD: "your-strong-root-password"
  MYSQL_PASSWORD: "your-strong-user-password"
```

> **安全建议**：生产环境请使用 [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets) 或 [External Secrets Operator](https://external-secrets.io/) 管理密钥，避免明文存储。

**2. 配置域名**

编辑 `k8s/base/ingress.yaml`，将 `docmind.example.com` 替换为你的实际域名。

**3. 构建并推送镜像**

```bash
# 构建镜像
docker build -t your-registry/docmind-backend:v1.0.0 ./backend
docker build -t your-registry/docmind-frontend:v1.0.0 ./frontend

# 推送到镜像仓库
docker push your-registry/docmind-backend:v1.0.0
docker push your-registry/docmind-frontend:v1.0.0
```

**4. 更新镜像引用**

编辑 `k8s/base/kustomization.yaml`：

```yaml
images:
  - name: docmind-backend
    newName: your-registry/docmind-backend
    newTag: v2.0.0
  - name: docmind-frontend
    newName: your-registry/docmind-frontend
    newTag: v2.0.0
```

**5. 部署**

```bash
# 开发环境
kubectl apply -k k8s/base/

# 生产环境（更多副本、更大资源、cert-manager TLS）
kubectl apply -k k8s/overlays/production/

# 或使用部署脚本
chmod +x k8s/deploy.sh
./k8s/deploy.sh deploy dev    # 开发环境
./k8s/deploy.sh deploy prod   # 生产环境
```

**6. 验证**

```bash
kubectl get all -n docmind
kubectl logs -f deployment/backend -n docmind
```

#### 高可用设计

| 特性 | 实现方式 | 说明 |
|------|----------|------|
| **多副本** | Backend 2-16 副本，Frontend 2-4 副本 | HPA 基于 CPU/Memory 自动扩缩 |
| **Pod 反亲和** | `podAntiAffinity: preferred` | 同一组件的 Pod 优先调度到不同节点 |
| **拓扑分布** | `topologySpreadConstraints` | 跨可用区均匀分布，单区故障不影响服务 |
| **PodDisruptionBudget** | `minAvailable: 1` | 自愿中断（维护/升级）期间至少保留 1 个 Pod |
| **多级探针** | startupProbe + livenessProbe + readinessProbe | 启动期 5 分钟宽限，运行期 30 秒检测，就绪期 10 秒检测 |

#### 零停机滚动更新

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1         # 先创建 1 个新 Pod
    maxUnavailable: 0   # 始终不低于期望副本数
```

更新流程：

1. 新 Pod 创建（Surge），旧 Pod 继续服务
2. 新 Pod 通过 `readinessProbe` → 加入 Service Endpoints
3. 旧 Pod 收到 SIGTERM → 优雅关闭（60s grace period）
4. 旧 Pod 从 Service Endpoints 移除 → 终止
5. 重复直到所有 Pod 更新完毕

> 配合 `PDB minAvailable=1`，即使节点维护也能保证服务连续性。

#### 数据持久化

| PVC | 容量 | 访问模式 | 挂载路径 | 用途 |
|-----|------|----------|----------|------|
| `rag-storage` | 50Gi | ReadWriteMany | `/app/rag_storage` | LightRAG 知识图谱 + 向量索引 |
| `output` | 20Gi | ReadWriteMany | `/app/output` | MinerU 解析输出 |
| `uploads` | 10Gi | ReadWriteMany | `/app/uploads` | 用户上传文档 |
| `logs` | 5Gi | ReadWriteMany | `/app/logs` | 应用日志 |
| `mysql-data` | 20Gi | ReadWriteOnce | `/var/lib/mysql` | MySQL 数据（StatefulSet 管理） |

> **重要**：Backend 的 PVC 使用 `ReadWriteMany`，要求 StorageClass 支持多节点同时读写。推荐方案：
> - **Longhorn**（开源，Kubernetes 原生，RWX 支持）
> - **NFS Server Provisioner**（简单，适合中小规模）
> - **AWS EFS / GCP Filestore**（云托管，生产级）
>
> 如果只有 `ReadWriteOnce` 的 StorageClass，需将 Backend 副本数设为 1，或使用 NFS sidecar 模式。

#### 自动扩缩容（HPA）

| 组件 | 最小副本 | 最大副本 | 扩缩指标 | 稳定窗口 |
|------|----------|----------|----------|----------|
| Backend | 2 | 16 (prod: 16) | CPU 70% / Memory 80% | 扩 60s / 缩 300s |
| Frontend | 2 | 4 | CPU 80% | 扩 60s / 缩 300s |

```bash
# 查看 HPA 状态
kubectl get hpa -n docmind

# 查看扩缩事件
kubectl describe hpa backend-hpa -n docmind
```

#### 网络安全（NetworkPolicy）

默认拒绝所有入站流量，仅开放必要路径：

| 组件 | 允许来源 | 端口 |
|------|----------|------|
| Frontend | Ingress Controller 命名空间 | 80 |
| Backend | Frontend Pod + Ingress Controller | 8000 |
| MySQL | Backend Pod | 3306 |

> 需要支持 NetworkPolicy 的 CNI 插件（Calico / Cilium / Weave 等）。

#### 安全加固

- **非 Root 运行**：所有容器以非 root 用户运行（Backend UID 1000，Frontend UID 1001，MySQL UID 999）
- **最小权限**：`capabilities: drop: [ALL]`，`allowPrivilegeEscalation: false`
- **网络隔离**：NetworkPolicy 默认拒绝，最小化攻击面
- **密钥管理**：Secret 独立于 ConfigMap，生产环境推荐 Sealed Secrets

#### 灾难恢复

**MySQL 备份**：

```bash
# 手动备份
kubectl exec mysql-0 -n docmind -- \
  mysqldump -u root -p$MYSQL_ROOT_PASSWORD \
  --single-transaction --routines --triggers docmind > backup.sql

# 使用部署脚本
./k8s/deploy.sh backup
```

**PVC 数据备份**（推荐 [Velero](https://velero.io/)）：

```bash
# 安装 Velero
velero install --provider aws --bucket velero-backups --secret-file ./credentials

# 创建定时备份
velero schedule create docmind-daily \
  --schedule="0 2 * * *" \
  --include-namespaces docmind \
  --ttl 720h

# 灾难恢复
velero restore create --from-schedule docmind-daily
```

**生产级 MySQL 高可用**（可选）：

单副本 MySQL 适合中小规模。如需自动故障转移，推荐：
- **MySQL InnoDB Cluster**（官方，3 节点，Group Replication + MySQL Router）
- **Vitess**（PlanetScale 开源，水平分片 + 自动故障转移）
- **AWS RDS Multi-AZ** / **阿里云 RDS 高可用版**（托管方案）

#### Kustomize Overlay 说明

| Overlay | 副本数 | Backend 资源 | MySQL 资源 | 用途 |
|---------|--------|-------------|-----------|------|
| `base` (dev) | Backend 2 / Frontend 2 / MySQL 1 | 500m CPU / 1Gi MEM | 250m / 512Mi | 开发测试 |
| `production` | Backend 3 / Frontend 3 / MySQL 1 | 1 CPU / 2Gi MEM | 500m / 1Gi | 生产环境 |

生产 Overlay 额外配置：
- Backend HPA 最大副本提升至 16
- MySQL `innodb_buffer_pool_size` 提升至 1G，`max_connections` 提升至 1000
- 启用 cert-manager 自动 TLS 证书签发

#### 常用运维命令

```bash
# 查看所有资源状态
kubectl get all,pvc,hpa,ingress -n docmind

# 查看滚动更新状态
kubectl rollout status deployment/backend -n docmind

# 查看特定 Pod 日志
kubectl logs -f deployment/backend -n docmind --tail=200

# 进入 Backend Pod 调试
kubectl exec -it deployment/backend -n docmind -- /bin/bash

# 手动扩缩容
kubectl scale deployment/backend --replicas=4 -n docmind

# 回滚到上一版本
kubectl rollout undo deployment/backend -n docmind

# 查看事件（排查调度问题）
kubectl get events -n docmind --sort-by='.lastTimestamp'

# 检查 PVC 绑定状态
kubectl get pvc -n docmind -o wide
```
---


## 项目结构

```
DocMind/
├── backend/                        # 后端 (FastAPI)
│   ├── main.py                     # 应用入口：FastAPI 实例、中间件栈、路由注册、MCP 挂载
│   ├── api/                        # API 层
│   │   ├── chat.py                 #   对话 API：KB/Direct 模式、SSE 流式、多会话管理
│   │   ├── documents.py            #   文档管理 API：上传、处理进度、列表、删除
│   │   ├── memory.py               #   记忆 API：CRUD、开关、知识提取配置
│   │   ├── ocr.py                  #   OCR API：PDF/图片/PPT → Tesseract + LLM 纠错
│   │   └── search.py               #   搜索 API：多引擎搜索 → 深度抓取 → LLM 综合
│   ├── core/                       # 核心业务层
│   │   ├── raganything.py          #   RAG-Anything 封装：文档解析 + 多模态 + 查询
│   │   ├── context_manager.py      #   上下文管理器 v2：分层摘要 + 指令持久化 + 去重
│   │   ├── agentic_retrieve.py     #   Agentic 检索：PageIndex 风格工具调用 Agent
│   │   ├── tree_index.py           #   树形文档索引：层级结构树，按页/按节点检索
│   │   ├── adaptive_retrieval.py   #   自适应检索：RF-Mem 熟悉度评估 + 双路径切换
│   │   ├── mcp_server.py           #   MCP Server：7 Tools + 2 Resources + 1 Prompt
│   │   ├── doc_handler.py          #   文档处理器：Upload → MinerU → LightRAG 管道
│   │   ├── batch_processor.py      #   批量处理器：Semaphore 并发 + 进度追踪 + 重试
│   │   ├── chat_history.py         #   对话历史：aiomysql 异步连接池
│   │   ├── memory.py               #   记忆服务：LLM 知识提取 + 语义索引 + SQLite
│   │   ├── meta_store.py           #   元数据存储：SQLite WAL 模式
│   │   ├── context_extractor.py    #   上下文提取器：页级/块级上下文窗口
│   │   ├── modal_processors.py     #   多模态处理器：图片/表格/公式 VLM 分析
│   │   ├── ocr_handler.py          #   OCR 处理器：Tesseract + LLM 纠错管道
│   │   ├── parser.py               #   文档解析器：PyMuPDF + LLM TOC 检测
│   │   ├── progress.py             #   进度追踪器：SQLite 持久化
│   │   ├── prompt_templates.py     #   Prompt 模板注册中心
│   │   └── web_scraper.py          #   网页抓取引擎：多引擎策略 + 自动降级
│   ├── infrastructure/             # 基础设施层
│   │   ├── llm_client.py           #   LLM 客户端：Tenacity 重试 + Fallback + Token 计费
│   │   ├── reranker.py             #   Cross-Encoder 重排序 + 三明治重排
│   │   ├── token_counter.py        #   Token 计数器：tiktoken 精确 + 字符启发式
│   │   ├── tracing.py              #   链路追踪：OpenTelemetry 风格 Trace/Span
│   │   ├── state_db.py             #   SQLite 持久化状态层
│   │   ├── errors.py               #   全局异常处理 + Correlation ID
│   │   ├── logging.py              #   结构化 JSON 日志
│   │   ├── response.py             #   标准化 API 响应
│   │   ├── security.py             #   安全中间件：API Key + 响应头 + IP 白/黑名单
│   │   └── validation.py           #   请求验证 + 滑动窗口限流
│   ├── config/
│   │   └── settings.py             # 全局配置：LLM、数据库、存储路径、MCP
│   ├── Dockerfile                  # 后端容器构建
│   └── .env.example                # 环境变量模板
│
├── frontend/                       # 前端 (React 18 + Ant Design 5)
│   ├── src/
│   │   ├── main.tsx                # 入口：BrowserRouter + Ant Design 中文国际化
│   │   ├── App.tsx                 # 路由定义：5 个页面
│   │   ├── index.css               # 设计系统 v2：CSS 变量体系 + Ant Design 覆盖
│   │   ├── layouts/
│   │   │   ├── MainLayout.tsx      # 主布局：可折叠 Sider + Header + Content
│   │   │   └── MainLayout.css      # 布局样式
│   │   ├── pages/
│   │   │   ├── Chat/               # 智能对话：KB/Direct 模式、SSE 流式、Markdown+KaTeX
│   │   │   ├── KnowledgeBase/      # 知识库管理：上传、3 种处理模式、批量导入
│   │   │   ├── Search/             # 深度搜索：多引擎 → 抓取 → AI 分析
│   │   │   ├── Memory/             # 记忆管理：4 种类型、开关、配置
│   │   │   └── OCR/                # OCR 识别：Tesseract + LLM 纠错
│   │   ├── components/             # 共享组件
│   │   │   ├── BatchImportModal.tsx#   批量导入弹窗：5 步向导
│   │   │   ├── ErrorBoundary.tsx   #   错误边界
│   │   │   ├── StatusTag.tsx       #   状态标签
│   │   │   ├── EmptyState.tsx      #   空状态占位
│   │   │   └── LoadingSpinner.tsx  #   加载指示器
│   │   ├── services/
│   │   │   └── api.ts              # API 客户端：类型安全 + Correlation ID + SSE 解析
│   │   └── types/
│   │       └── api.ts              # TypeScript 类型定义（与后端 Pydantic 对齐）
│   ├── Dockerfile                  # 前端多阶段构建：node:20 → nginx
│   ├── nginx.conf                  # Nginx 配置：gzip + SPA fallback + API 代理
│   ├── vite.config.ts              # Vite 构建配置
│   └── tailwind.config.js          # Tailwind CSS 配置
│
├── docker-compose.yml              # 全栈部署编排
└── README.md                       # 本文件
```

---

## API 参考

### 文档管理

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/documents/upload` | 上传文档，返回 `doc_id` |
| `POST` | `/api/documents/{doc_id}/process?mode=fast\|standard\|full` | 触发文档处理 |
| `GET` | `/api/documents/list` | 获取文档列表 |
| `GET` | `/api/documents/{doc_id}/progress` | 查询处理进度 |
| `DELETE` | `/api/documents/{doc_id}` | 删除文档 |
| `POST` | `/api/documents/batch/scan?directory=...` | 扫描目录 |
| `POST` | `/api/documents/batch/start?directory=...&mode=...` | 启动批量处理 |

### 对话

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/chat/stream` | SSE 流式对话（KB/Direct 模式） |
| `GET` | `/api/chat/sessions` | 获取会话列表 |
| `GET` | `/api/chat/sessions/{session_id}` | 获取会话消息 |
| `DELETE` | `/api/chat/sessions/{session_id}` | 删除会话 |
| `POST` | `/api/chat/stop` | 中断当前生成 |

### 搜索

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/search/results` | 多引擎搜索，返回 URL 列表 |
| `POST` | `/api/search/scrape-page` | 深度抓取单个页面 + AI 分析 |

### 记忆

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/memory/entries` | 获取记忆条目（支持分页、筛选） |
| `PUT` | `/api/memory/entries/{id}/enable` | 启用记忆 |
| `PUT` | `/api/memory/entries/{id}/disable` | 禁用记忆 |
| `DELETE` | `/api/memory/entries/{id}` | 删除记忆 |
| `GET` | `/api/memory/config` | 获取记忆配置 |
| `PUT` | `/api/memory/config` | 更新记忆配置 |

### OCR

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/ocr/process-poll` | 上传文件并启动 OCR |
| `GET` | `/api/ocr/progress/{doc_id}` | 查询 OCR 进度 |
| `GET` | `/api/ocr/result/{doc_id}` | 获取 OCR 结果 |
| `POST` | `/api/ocr/correct/{doc_id}` | 启动 LLM 智能纠错 |

### MCP

| 端点 | 说明 |
|------|------|
| `POST /mcp` | MCP JSON-RPC 端点（Streamable HTTP） |

---

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `GEMINI_API_KEY` | 是 | — | LLM API 密钥 |
| `GEMINI_BASE_URL` | 否 | `https://generativelanguage.googleapis.com/v1beta/openai/` | LLM API 地址 |
| `GEMINI_MODEL` | 否 | `gemini-2.5-flash` | LLM 模型名 |
| `EMBEDDING_API_KEY` | 否 | 同 `GEMINI_API_KEY` | Embedding API 密钥 |
| `EMBEDDING_MODEL` | 否 | `text-embedding-3-small` | Embedding 模型 |
| `MYSQL_HOST` | 是 | `localhost` | MySQL 地址 |
| `MYSQL_PORT` | 否 | `3306` | MySQL 端口 |
| `MYSQL_PASSWORD` | 是 | — | MySQL 密码 |
| `MYSQL_DATABASE` | 否 | `docmind` | MySQL 数据库名 |
| `TESSERACT_PATH` | 否 | — | Tesseract 可执行文件路径 |
| `MCP_ENABLED` | 否 | `true` | 是否启用 MCP Server |
| `API_KEYS` | 否 | — | API Key 认证（逗号分隔，为空则禁用） |
| `ENVIRONMENT` | 否 | `development` | 运行环境（`development` / `production`） |

---

## 部署指南

### 本地开发

```bash
# 后端（热重载）
cd backend && uvicorn main:app --reload --port 8000

# 前端（热重载）
cd frontend && npm run dev
```

### Docker Compose（推荐生产部署）

```bash
docker compose up -d
```

### 可选：本地 Ollama

取消 `docker-compose.yml` 中 `ollama` 服务的注释，配合 NVIDIA Container Toolkit 使用 GPU 加速。后端环境变量指向 `http://ollama:11434`。

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **前端框架** | React 18 + TypeScript | UI 构建 |
| **UI 组件库** | Ant Design v5 | 企业级组件 |
| **构建工具** | Vite 6 | 开发/构建 |
| **样式方案** | CSS Modules + CSS Variables + Tailwind | 模块化样式 + 设计系统 |
| **Markdown** | react-markdown + remark-math + rehype-katex | 公式渲染 |
| **后端框架** | FastAPI + asyncio | 异步 API |
| **RAG 引擎** | RAG-Anything + LightRAG | 文档解析 + KG + 向量 |
| **文档解析** | MinerU | PDF/DOCX/PPTX 解析 |
| **OCR** | Tesseract + LLM 纠错 | 文字提取 + 智能纠错 |
| **向量存储** | NanoVQ | 轻量级向量数据库 |
| **知识图谱** | GraphML | 实体 + 关系存储 |
| **数据库** | MySQL 8.0 | 会话历史、文档元数据 |
| **状态存储** | SQLite (WAL) | 进度、记忆、元数据 |
| **LLM 客户端** | OpenAI SDK + Tenacity | 重试 + Fallback |
| **Token 计数** | tiktoken | 精确 token 计数 |
| **重排序** | Cross-Encoder (Cohere) | 精确相关性评分 |
| **MCP** | mcp Python SDK (FastMCP) | 标准化工具暴露 |
| **容器化** | Docker + Docker Compose | 全栈部署 |
| **Web 服务器** | Nginx | 前端静态资源 + API 代理 |

---

## 设计决策

### 为什么不用 ChromaDB？

ChromaDB 在 Windows 上存在兼容性问题，且对大规模文档的内存占用较高。我们选择了 **NanoVQ**（轻量级向量存储）+ **PageIndex 风格的树形索引**（`tree_index.py`），每个文档存为 JSON，包含层级结构树，支持按页/按节点检索，零外部依赖。

### 为什么 MCP 用 Streamable HTTP 而不是 stdio？

stdio 适合本地单进程场景，但 DocMind 是 Web 服务。Streamable HTTP 让任何远程 MCP 客户端都能连接，且 `stateless_http=True` 模式支持多实例水平扩展。

### 为什么 ContextManager 不直接扩到 128k？

实测表明，128k 窗口下延迟翻倍但回答质量无提升。根本原因不是窗口不够大，而是**信息密度不够高**。ContextManager v2 通过分层压缩保持恒定信息密度，32k 窗口即可覆盖 50+ 轮对话。

---

## License

MIT License

---

## 致谢

DocMind 的诞生离不开以下开源项目的卓越贡献，在此向所有作者和社区成员致以诚挚的感谢：

### [RAG-Anything](https://github.com/hkuds/rag-anything)

RAG-Anything 是一个多模态 RAG 框架，为 DocMind 提供了核心的文档解析与多模态处理能力。它封装了 MinerU 文档解析引擎，支持 PDF、DOCX、PPTX 等多种格式的精确提取，并内置了图片、表格、公式的多模态分析管线。DocMind 的三档处理模式（Fast/Standard/Full）和多模态内容处理器（`modal_processors.py`）均深度参考了 RAG-Anything 的架构设计。

### [LightRAG](https://github.com/HKUDS/LightRAG)

LightRAG 是香港大学数据科学实验室开发的轻量级 RAG 框架，为 DocMind 提供了知识图谱构建与多模式查询能力。其创新的实体-关系图谱提取算法和 local/global/hybrid/naive 四模式查询策略，是 DocMind RAG 引擎的基石。DocMind 的自适应检索（RF-Mem）和 Agentic 检索（PageIndex 风格）均是在 LightRAG 的存储层之上构建的扩展能力。
