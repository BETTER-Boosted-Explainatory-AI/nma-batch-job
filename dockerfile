FROM tensorflow/tensorflow:2.19.0

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader wordnet

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/local/nltk_data

COPY . .

CMD ["python", "main.py"]