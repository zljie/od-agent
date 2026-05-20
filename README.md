# OD Agent - Customer Service Agent

AI-powered customer service agent powered by AgentScope and DeepSeek.

## Features

- Customer service dialogue with AI
- Powered by DeepSeek LLM
- FastAPI-based HTTP API
- Deployable to Railway

## Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy and configure environment variables:

```bash
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

3. Run the application:

```bash
python -m src.app
# or
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Welcome page
- `GET /health` - Health check
- `POST /chat` - Chat endpoint
- `POST /process` - Process requests

## Deployment to Railway

1. Push code to GitHub:

```bash
git add .
git commit -m "feat: initial AgentScope customer service agent"
git push origin main
```

2. Connect your GitHub repository to Railway

3. Add environment variable: `DEEPSEEK_API_KEY`

4. Deploy!

## Testing

```bash
pytest tests/
```
