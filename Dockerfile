# 使用阿里云加速官方 Python 镜像
FROM python:3.9-slim

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 设置 NLTK 数据路径
ENV NLTK_DATA=/app/nltk_data

# 安装系统依赖（OpenCV 编译所需）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopencv-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# 创建虚拟环境（可选，但推荐）
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA')" && \
    python -c "import nltk; nltk.download('stopwords', download_dir='$NLTK_DATA')"

# 复制模型和配置文件
COPY textbook.json .
COPY academic_terms.txt .
COPY best_model_xgb.pkl .
COPY scaler.pkl .
COPY weights_xgb.pkl .

# 复制主程序
COPY textbook_difficulty_system.py .

# 指定入口
CMD ["python", "textbook_difficulty_system.py"]
