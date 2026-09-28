FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
