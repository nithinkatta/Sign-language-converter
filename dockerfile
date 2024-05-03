# Use the official Python image from the Docker Hub
FROM python:3.10.11 

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install -r requirements.txt
RUN apt update && apt install -y libsm6 libxext6 libgl1 
RUN apt-get install -y libxrender-dev

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 5000 to the outside world
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "flask__.py"]
