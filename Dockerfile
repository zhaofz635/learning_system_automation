FROM ubuntu:24.04

# 安装系统依赖和工具
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1 \
    libgtk2.0-dev \
    python3.9 \
    python3.9-venv \
    python3-pip \
    git \
    jq \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 将 Python 3.9 设置为默认 python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/python3.9-pip 1

# 升级 pip
RUN pip3 install --upgrade pip setuptools wheel

# 预安装 NLTK 并下载数据（避免运行时下载）
RUN pip3 install nltk \
    && mkdir -p /nltk_data \
    && python3 -c "import nltk; nltk.download('punkt', download_dir='/nltk_data'); nltk.download('punkt_tab', download_dir='/nltk_data'); nltk.download('stopwords', download_dir='/nltk_data');"

ENV NLTK_DATA=/nltk_data
