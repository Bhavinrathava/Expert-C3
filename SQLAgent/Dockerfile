# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev g++ \
    && rm -rf /var/lib/apt/lists/*
    
# Install any needed packages specified in requirements.txt
RUN pip --no-cache install uvicorn fastapi torch transformers accelerate

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV MODEL_NAME=defog/sqlcoder-7b-2

# Run app.py when the container launches
CMD ["uvicorn", "sql_API:app", "--host", "0.0.0.0", "--port", "8000"]
