# Use an appropriate base image (e.g., python:3.10-slim-buster or similar)
FROM python:3.13.0-slim

# Set the working directory
WORKDIR /app

# Install the missing libxcb1 and potentially other XCB dependencies
# The list below covers common requirements for GUI apps in headless environments
RUN apt-get update && apt-get install -y \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-util1 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    # Add other potential dependencies here if needed (e.g., libgl1-mesa-glx for OpenGL)
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy your application code into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run your application (replace with your actual command)
CMD ["python", "your_app_file.py"]
