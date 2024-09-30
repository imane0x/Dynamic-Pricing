# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install necessary dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn pandas numpy wandb

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
