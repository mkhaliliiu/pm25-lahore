# Dashboard and API image. Build from the repository root:
#   docker build -t pm25-lahore .
#   docker run -p 8501:8501 pm25-lahore                                   # Streamlit dashboard
#   docker run -p 8000:8000 pm25-lahore uvicorn api.main:app --host 0.0.0.0 --port 8000   # FastAPI
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY src src
COPY app app
COPY api api
COPY models models
COPY results results

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
