FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
