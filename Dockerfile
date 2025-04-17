FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all app files
COPY . .

# Expose the port Flask will run on
EXPOSE 7860

# Run your Flask app
CMD ["python", "app.py"]
