FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY sandbox/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
COPY backend/app/task_templates /usr/local/lib/python3.12/site-packages/ml_tasks
COPY sandbox/ml_runtime.py /usr/local/lib/python3.12/site-packages/ml_runtime.py
COPY sandbox/experiment_export.py /usr/local/lib/python3.12/site-packages/experiment_export.py
USER 10001:10001
WORKDIR /work
CMD ["sleep", "infinity"]
