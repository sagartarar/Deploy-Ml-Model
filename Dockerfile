# Dockerfile

# 1. Base Image: Use an official Python slim image
FROM python:3.9-slim

# 2. Set Environment Variables (optional but good practice)
ENV PYTHONDONTWRITEBYTECODE=1 
#Prevents python from writing .pyc files
ENV PYTHONUNBUFFERED=1      
#Force stdout/stderr streams to be unbuffered

# 3. Set Working Directory
WORKDIR /app

# 4. Install System Dependencies (if any)
# Example: RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 6. Install Python Dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy Application Code and Model
# Copy the application directory contents into /app/app
COPY ./app /app/app
# Copy the model directory contents into /app/model
COPY ./model /app/model

# 8. Expose Port the application runs on
EXPOSE 8000

# 9. Command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
# The list format is preferred for CMD
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]