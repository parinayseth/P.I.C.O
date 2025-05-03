FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv python3-tk curl && \
    pip3 install --upgrade pip --break-system-packages

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml poetry.lock /app/

# Install dependencies (with Poetry-managed virtualenv)
RUN poetry install --no-root --no-interaction --no-ansi

# Copy full source after dependencies
COPY . /app

# Expose port
EXPOSE 5005

# Run using Poetry's venv path
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "4"]
