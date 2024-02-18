FROM python:3.10.7-slim-bullseye as base
USER root
RUN  apt-get update && \
        apt-get install -y \
        build-essential \
        gcc \
        libz-dev \
        bedtools \
        bwa \
        samtools

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
COPY ./src ./src
COPY ./analysis ./analysis
COPY ./main.py ./main.py
ENTRYPOINT ["python", "main.py", "sim", "--new_data", "--chr", "all"]