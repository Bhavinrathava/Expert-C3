FROM python:3.9-slim

WORKDIR /app

COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "MongoAPI:app", "--host", "0.0.0.0", "--port", "8000"]

