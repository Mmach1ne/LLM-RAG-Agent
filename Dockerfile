FROM python:3.11-slim

# install Rust toolchain
RUN apt-get update \
 && apt-get install -y curl build-essential \
 && curl https://sh.rustup.rs -sSf | bash -s -- -y \
 && . $HOME/.cargo/env \
 && rm -rf /var/lib/apt/lists/*

# put Cargo cache inside the image
ENV CARGO_HOME=/root/.cargo
ENV CARGO_TARGET_DIR=/root/.cargo/target

WORKDIR /app
COPY ./backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend ./backend
CMD ["uvicorn", "backend.fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000"]
