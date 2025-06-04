FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader wordnet

# Copy the rest of your code
COPY . .

# Set environment variables (optional, for unbuffered output)
ENV PYTHONUNBUFFERED=1

# Default command to run your main.py
CMD ["python", "main.py"]