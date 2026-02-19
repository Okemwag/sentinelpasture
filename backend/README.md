# Backend API Server

FastAPI backend that connects the Next.js dashboard to the Python AI Engine.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check
- `GET /` - Service status
- `GET /health` - Health check with AI engine status

### AI Processing
- `POST /api/process` - Process signals through AI engine
- `GET /api/predict` - Generate risk predictions
- `POST /api/analyze-drivers` - Analyze risk drivers
- `POST /api/recommend-interventions` - Get intervention recommendations

## Configuration

The backend automatically detects if the AI engine is available. If not, it uses mock data for development.

To use the real AI engine:
1. Ensure `ai-engine` is installed
2. The backend will automatically import and use it

## Development Mode

The backend runs in mock mode by default, which is perfect for frontend development without needing the full AI engine running.

## Production Deployment

For production:
1. Install the AI engine dependencies
2. Configure environment variables
3. Use a production ASGI server like Gunicorn with Uvicorn workers

```bash
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
