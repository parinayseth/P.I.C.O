FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# Install Python, pip, and system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv python3-tk curl && \
    pip3 install --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy only pyproject and lock file first to cache dependency install
COPY pyproject.toml poetry.lock /app/

# Install dependencies (without creating a virtualenv inside the container)
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Copy the rest of the code
COPY . /app

EXPOSE 5005

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "4"]
