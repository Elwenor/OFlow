FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[api,http]' && useradd --create-home --uid 10001 oflow
COPY configs ./configs
USER oflow
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"
CMD ["oflow", "serve", "--config", "configs/api.yaml", "--host", "0.0.0.0"]
