FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim

RUN apt-get update && apt-get install -y python3-tk || true

WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt --break-system-packages

EXPOSE 5005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "4"]