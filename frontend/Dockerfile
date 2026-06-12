# ===== Stage 1: Builder =====
# 注意：必须装 devDependencies（typescript、vite 在 build 阶段要用）
FROM node:20.18-alpine AS builder

WORKDIR /app

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
FROM nginx:1.27.3-alpine

# 清理 nginx 默认配置，把我们自己的塞进去
RUN rm -f /etc/nginx/conf.d/default.conf

# 复制 build 产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制 nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# 健康检查：能 200 拿到首页就算 OK
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ > /dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
