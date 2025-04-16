FROM python:3.10-slim-bookworm

WORKDIR /leapdraw

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .
COPY categories.txt .
COPY data.py .
COPY model.h5 .
COPY Backend.py .

CMD ["uvicorn", "Backend:app", "--host", "0.0.0.0", "--port", "8000"]