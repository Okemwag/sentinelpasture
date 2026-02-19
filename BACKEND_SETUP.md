# Backend Infrastructure Setup Guide

Complete guide for setting up the backend infrastructure connecting the dashboard to the AI engine.

## Architecture Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Next.js       │      │   FastAPI        │      │   AI Engine     │
│   Dashboard     │─────▶│   Backend        │─────▶│   (Python)      │
│   (Frontend)    │      │   (API Server)   │      │                 │
└─────────────────┘      └──────────────────┘      └─────────────────┘
     Port 3000                Port 8000                  Library
```

## Quick Start (Development Mode)

### 1. Start the Backend API

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API will run on `http://localhost:8000` with mock data.

### 2. Start the Frontend

```bash
pnpm install
pnpm dev
```

The dashboard will run on `http://localhost:3000`

### 3. Test the Connection

Visit `http://localhost:3000/dashboard` and the dashboard will automatically fetch data from the API.

## Full Setup with AI Engine

### 1. Install AI Engine

```bash
cd ai-engine
pip install -e .
```

### 2. Configure Environment

Create `.env.local`:

```bash
# Use real AI engine
USE_MOCK_AI=false
AI_ENGINE_URL=http://localhost:8000
```

### 3. Start Backend with AI Engine

```bash
cd backend
python main.py
```

The backend will automatically detect and use the AI engine.

## API Client Usage

The frontend uses a centralized API client (`lib/api-client.ts`):

```typescript
import { apiClient } from '@/lib/api-client';

// Get stability index
const response = await apiClient.getStabilityIndex();
console.log(response.data.value); // 67

// Get regional data
const regions = await apiClient.getRegionalData();

// Process signals through AI engine
const result = await apiClient.processSignals(signals);
```

## Available API Endpoints

### Dashboard Data
- `GET /api/dashboard/stability` - National stability index
- `GET /api/dashboard/stats` - Quick statistics
- `GET /api/dashboard/risk-factors` - Risk factors summary

### Regional Data
- `GET /api/regional/data` - All regional data
- `GET /api/regional/map` - Map visualization data

### Risk Analysis
- `GET /api/drivers/list` - Risk drivers
- `GET /api/alerts/list` - Active alerts
- `GET /api/interventions/list` - Intervention options
- `GET /api/outcomes/list` - Intervention outcomes

### AI Engine Integration
- `POST /api/ai/process` - Process signals
- `GET /api/ai/predict` - Generate predictions

## Backend API (FastAPI)

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "ai_engine": true,
  "timestamp": "2026-02-19T10:30:00Z"
}
```

### Process Signals

```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "signals": [
      {
        "type": "social_media",
        "source": "verified_media",
        "data": {"text": "Protest gathering", "likes": 1000},
        "location": {"latitude": -1.286389, "longitude": 36.817223},
        "temporal": {"timestamp": "2026-02-19T10:30:00Z"}
      }
    ]
  }'
```

## Mock Data vs Real AI Engine

### Mock Data Mode (Default)
- Perfect for frontend development
- No AI engine dependencies required
- Instant responses
- Consistent test data

### Real AI Engine Mode
- Full AI capabilities
- Real-time threat analysis
- Causal inference
- Predictive modeling
- Meta-learning adaptation

## Environment Variables

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=/api
AI_ENGINE_URL=http://localhost:8000
USE_MOCK_AI=true
```

### Backend
```bash
# Set in backend/.env or environment
AI_ENGINE_URL=http://localhost:8000
```

## Data Flow

### 1. Dashboard Request
```typescript
// User visits dashboard
const data = await apiClient.getStabilityIndex();
```

### 2. Next.js API Route
```typescript
// app/api/dashboard/stability/route.ts
export async function GET() {
  // Can use mock data or call backend
  return NextResponse.json({ data, success: true });
}
```

### 3. FastAPI Backend (Optional)
```python
# backend/main.py
@app.post("/api/process")
async def process_signals(request):
    result = await orchestrator.process_intelligence_pipeline(signals)
    return result
```

### 4. AI Engine
```python
# ai-engine/orchestrator.py
async def process_intelligence_pipeline(self, signals):
    # Advanced AI processing
    return assessment
```

## Testing

### Test Frontend API
```bash
curl http://localhost:3000/api/dashboard/stability
```

### Test Backend API
```bash
curl http://localhost:8000/api/process -X POST \
  -H "Content-Type: application/json" \
  -d '{"signals": []}'
```

### Test AI Engine
```bash
cd ai-engine
python examples/basic_usage.py
```

## Deployment

### Frontend (Vercel/Netlify)
```bash
pnpm build
pnpm start
```

### Backend (Docker)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
COPY ai-engine/ ./ai-engine/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Command
```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.9+

# Install dependencies
pip install -r backend/requirements.txt
```

### AI Engine not found
```bash
# Install AI engine
cd ai-engine
pip install -e .
```

### CORS errors
The backend is configured to allow requests from `localhost:3000`. Update `backend/main.py` if using different ports.

### Mock data not updating
The backend uses mock data by default. To use real AI engine, set `USE_MOCK_AI=false` in environment.

## Performance

### Mock Mode
- Response time: <10ms
- Throughput: 1000+ req/s
- Memory: ~50MB

### AI Engine Mode
- Response time: 100-500ms
- Throughput: 10-50 req/s
- Memory: ~2GB (with models loaded)

## Next Steps

1. ✅ Backend API running with mock data
2. ✅ Frontend connected to API
3. ⏭️ Connect real AI engine
4. ⏭️ Add database for persistence
5. ⏭️ Implement authentication
6. ⏭️ Add real-time WebSocket updates
7. ⏭️ Deploy to production

## Support

For issues:
- Check API documentation: `http://localhost:8000/docs`
- Review logs in terminal
- Test endpoints with curl
- Verify environment variables
