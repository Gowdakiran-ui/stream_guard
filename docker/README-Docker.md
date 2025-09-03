# StreamGuard Fraud Detection - Docker Deployment

This guide explains how to deploy the StreamGuard Fraud Detection application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)
- Training data file: `train_transaction.csv`

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and start the application:**
```bash
docker-compose up --build
```

2. **Access the application:**
- Web UI: http://localhost:5000
- Demo: http://localhost:5000/demo

### Option 2: Using Docker directly

1. **Build the image:**
```bash
docker build -t streamguard-ml .
```

2. **Run the container:**
```bash
docker run -p 5000:5000 \
  -v $(pwd)/train_transaction.csv:/app/train_transaction.csv:ro \
  -v $(pwd)/models:/app/models \
  streamguard-ml
```

## Training Models

### Using Docker Compose
```bash
# Train models using the training profile
docker-compose --profile training up streamguard-trainer
```

### Using Docker directly
```bash
docker run --rm \
  -v $(pwd)/train_transaction.csv:/app/train_transaction.csv:ro \
  -v $(pwd)/models:/app/models \
  streamguard-ml train
```

## Volume Mounts

The application uses several volume mounts:

- `./train_transaction.csv:/app/train_transaction.csv:ro` - Training data (read-only)
- `./models:/app/models` - Trained model files
- `./data:/app/data` - Application data
- `./logs:/app/logs` - Application logs

## Environment Variables

- `FLASK_ENV` - Flask environment (production/development)
- `PYTHONUNBUFFERED` - Python output buffering

## Container Commands

The container supports different modes:

- `app` (default) - Start the web application
- `train` - Train ML models only
- Custom commands - Execute any command in the container

### Examples:

```bash
# Start web app (default)
docker run streamguard-ml

# Train models only
docker run streamguard-ml train

# Run custom command
docker run streamguard-ml python --version
```

## Health Checks

The container includes health checks that verify the application is running:
- Endpoint: http://localhost:5000/
- Interval: 30 seconds
- Timeout: 30 seconds
- Retries: 3

## Troubleshooting

### Models not found
If you see warnings about missing model files:
1. Ensure `train_transaction.csv` is mounted correctly
2. Run training: `docker-compose --profile training up streamguard-trainer`

### Port conflicts
If port 5000 is already in use:
```bash
# Use a different port
docker run -p 8080:5000 streamguard-ml
```

### Memory issues
For large datasets, increase Docker memory limits:
```bash
docker run --memory=4g streamguard-ml
```

## Production Deployment

For production deployment:

1. **Use environment-specific compose file:**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  streamguard-ml:
    build: .
    ports:
      - "80:5000"
    environment:
      - FLASK_ENV=production
    restart: always
```

2. **Deploy:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Monitoring

View container logs:
```bash
# Using Docker Compose
docker-compose logs -f streamguard-ml

# Using Docker directly
docker logs -f <container_name>
```

## Cleanup

Remove containers and images:
```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi streamguard-ml

# Remove volumes (careful - this deletes data)
docker-compose down -v
```
