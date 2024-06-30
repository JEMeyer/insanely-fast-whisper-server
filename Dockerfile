# Use an official PyTorch runtime with CUDA support as a parent image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install the insanely-fast-whisper-server package
RUN pip3 install .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "insanely-fast-whisper-server.app:app", "--host", "0.0.0.0", "--port", "8000"]
