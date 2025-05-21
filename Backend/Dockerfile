# Use official Python base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy all project files to the container's working directory
COPY . /app

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask default port
EXPOSE 5001

# Run the Flask backend
CMD ["python", "ml_project.py"]
