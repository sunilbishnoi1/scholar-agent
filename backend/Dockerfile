FROM python:3.11-slim

WORKDIR /app

# install system deps if needed
RUN apt-get update && apt-get install -y build-essential --no-install-recommends && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app files and ensure start.sh is executable
COPY . .
RUN chmod +x ./start.sh

ENV PORT=10000
EXPOSE 10000

# Use exec form to preserve signals
CMD ["./start.sh"]
