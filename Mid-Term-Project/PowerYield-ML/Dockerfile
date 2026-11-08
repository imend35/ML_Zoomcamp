
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY predict.py .
COPY templates/ templates/
COPY models/poweryield_model.json models/poweryield_model.json

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]