# ===== Stage 1: Builder =====
# 注意：必须装 devDependencies（typescript、vite 在 build 阶段要用）
FROM node:20.18-alpine AS builder

LABEL maintainer="DocMind Team" \
      version="2.0.0" \
      description="DocMind Frontend Builder"

WORKDIR /app

# 构建时环境变量（通过 docker-compose build args 注入）
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# 先 COPY package 文件走 layer cache
COPY package*.json ./

# 保留 npm cache，下次构建秒级
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# 复制源码并 build（package.json 里的 build = "tsc -b && vite build"）
COPY . .
RUN npm run build


# ===== Stage 2: Runtime =====
# nginx:alpine 自带 wget，没有 curl —— healthcheck 用 wget
FROM nginx:1.27.3-alpine AS runtime

LABEL maintainer="DocMind Team" \
      version="2.0.0" \
      description="DocMind Frontend - Enterprise Document Intelligence Platform"

# 清理 nginx 默认配置
RUN rm -f /etc/nginx/conf.d/default.conf \
    && rm -f /usr/share/nginx/html/*

# 复制 build 产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制 nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 创建非 root 用户（nginx 默认以 nginx 用户运行，但显式声明更安全）
RUN addgroup -g 1001 -S appgroup \
    && adduser -u 1001 -S appuser -G appgroup \
    && chown -R appuser:appgroup /usr/share/nginx/html \
    && chown -R appuser:appgroup /var/cache/nginx \
    && chown -R appuser:appgroup /var/log/nginx \
    && touch /var/run/nginx.pid \
    && chown -R appuser:appgroup /var/run/nginx.pid

# K8s: nginx listens on 8080 (non-root cannot bind port 80)
# Docker standalone: nginx listens on 80 (default)
# The K8s ConfigMap overrides the server block to listen on 8080
EXPOSE 80

# 健康检查：使用 wget 检查端口 80（Docker standalone 场景）
# K8s 部署时使用自己的 livenessProbe/readinessProbe，不依赖此 HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
