# 1. 使用官方 Python 3.9 镜像
FROM python:3.9

# 2. 设置工作目录
WORKDIR /app

# 3. 复制依赖清单并安装
# (先复制 requirements.txt 利用 Docker 缓存加速构建)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制后端代码文件夹
# 注意：这会把本地 backend/server 复制到容器的 /app/server
COPY server/ ./server/

# 5. 创建非 root 用户 (Hugging Face 安全要求)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 6. 启动命令
# 必须监听 7860 端口，且 host 为 0.0.0.0
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]