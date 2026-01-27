# Use an official lightweight Python image.
# 3.12 matches the user's local environment
FROM python:3.12-slim

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for RDKit and other libraries
# libxrender1 and libxext6 are common requirements for RDKit's graphical components
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Expose the port the app runs on
EXPOSE 3001

# Create a non-root user to run the application
# This is a critical security best practice to limit the impact of potential vulnerabilities
RUN addgroup --system appgroup && adduser --system --group appuser

# Change ownership of the application directory to the non-root user
RUN chown -R appuser:appgroup /code

# Switch to the non-root user
USER appuser

# Command to run the application
# We use 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3001"]
