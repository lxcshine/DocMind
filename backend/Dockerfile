# ===== Stage 1: Builder =====
# 装编译工具链，build 出可安装的 Python 包 wheel 缓存到 /install
FROM python:3.11.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 仅 builder 需要的编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先 COPY requirements 走 Docker layer cache（业务代码变动不重装依赖）
COPY requirements.txt .
# BuildKit cache mount：保留 pip 包缓存，下次构建秒级
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt


# ===== Stage 2: Runtime =====
# 干净的 slim 镜像 + runtime 系统库，体积小
FROM python:3.11.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 运行时必需的系统库 —— 之前漏装会导致 PDF/OCR/图像处理全部失败
#   - poppler-utils:        pdf2image 依赖
#   - tesseract + chi_sim:  pytesseract OCR（settings.OCR_LANGUAGES=chi_sim+eng）
#   - libgl1/libglib/libsm/libxrender/libxext:  Pillow / pymupdf / 视觉库
#   - fonts-noto-cjk:       中文渲染（matplotlib、KaTeX 文本）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 非 root 用户
RUN groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# 从 builder 复制已安装的 Python 包
COPY --from=builder /install /usr/local

# 复制应用代码（chown 让 appuser 可写）
COPY --chown=appuser:appuser . .

# 运行时数据目录
RUN mkdir -p /app/logs /app/uploads /app/rag_storage /app/output \
    && chown -R appuser:appuser /app/logs /app/uploads /app/rag_storage /app/output

USER appuser

EXPOSE 8000

# start_period=300s 给 RAG 初始化（首次加载模型权重）留时间
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 不指定 --workers：默认就是 1，配合 FastAPI async 处理 SSE 长连接
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# ⚠️  MinerU（magic-pdf）说明
#   requirements.txt 里 magic-pdf 是注释掉的（依赖太重：PaddlePaddle +
#   PaddleOCR + torch + 模型权重 ~4GB）。如果生产环境确实要用 MinerU 解析：
#
#   1) 取消 requirements.txt 末尾 magic-pdf 的注释
#   2) 在 runtime 阶段额外安装：
#        libreoffice  libpq-dev
#   3) 用独立 GPU 服务跑（参考 docker-compose.yml 注释里的 ollama 块）
#   4) settings.MINERU_BACKEND 设为 pipeline（避免 hybrid-auto-engine
#      在低显存下的内部超时）
# ============================================================================
