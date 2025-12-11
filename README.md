# Multi-Modal Voice Agent Pipeline

A production-ready voice agent system that combines speech-to-text, LLM tool calling, and text-to-speech into a single async pipeline. Built for low latency and high concurrency.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Does

- Accepts text or voice input via WebSocket
- Processes through OpenAI GPT-4 with custom tools
- Generates interactive visualizations with Plotly
- Returns synchronized audio and visual responses
- Tracks costs and performance metrics in real-time

**Performance**: 2-4 second average response time, supports 50+ concurrent sessions

## Quick Start

### Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [ElevenLabs API key](https://elevenlabs.io/app/settings/api-keys)

### Installation

```bash
git clone https://github.com/Parakrant/multimodal-voice-agent.git
cd multimodal-voice-agent
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

### Run

```bash
python main_enhanced.py
```

Then open `client_enhanced.html` in your browser and click "Connect".

## Architecture

The pipeline processes each request through five async stages:

1. **STT** (500-800ms) - ElevenLabs speech-to-text
2. **LLM** (1-2s) - OpenAI GPT-4 with tool calling
3. **Tools** (100-500ms) - Chart generation, sentiment analysis, data fetching
4. **Visualization** (100-200ms) - Plotly chart generation
5. **TTS** (800-1200ms) - ElevenLabs text-to-speech

All stages run asynchronously with connection pooling and rate limiting.

## API Endpoints

**WebSocket**
- `/ws/multimodal` - Main voice/text interaction endpoint

**REST**
- `GET /health` - Server status
- `GET /metrics` - Performance and cost metrics
- `GET /session/{id}` - Session details
- `GET /data-viz/{id}` - Visualization data

## Usage Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/multimodal');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'user_text',
        text: 'Create a bar chart with values 10, 20, 30, 40'
    }));
};
```

## Available Tools

The LLM can call three built-in tools:

1. **Chart Generation** - Creates line, bar, pie, or scatter charts
2. **Sentiment Analysis** - Analyzes text sentiment with scoring
3. **Real-Time Data** - Fetches system metrics and simulated data

## Configuration

**Adjust concurrency for your ElevenLabs tier** (line 92 in `main_enhanced.py`):
```python
ELEVENLABS_SEMAPHORE = asyncio.Semaphore(3)  # Free/Starter: 3, Creator: 10
```

**Switch LLM model** (line 74):
```python
OPENAI_MODEL = "gpt-3.5-turbo"  # Faster and 90% cheaper
```

## Cost Breakdown

Typical cost per conversation turn: **~$0.076**
- LLM: $0.016 (65%)
- TTS: $0.048 (30%)
- STT: $0.012 (5%)

**Reduce costs:**
- Use GPT-3.5-Turbo instead of GPT-4
- Limit max_tokens in responses
- Reduce conversation history from 10 to 5 turns
- Cache common responses

## Testing

```bash
# Run benchmarks
python benchmark.py --max-concurrency 10

# Check health
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics
```

## Documentation

- [Quick Start Guide](QUICKSTART.md) - Get running in 5 minutes
- [Rate Limiting Guide](RATE_LIMITING_FIX.md) - Handle API limits
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

## How It Works

Each WebSocket connection creates a session that maintains:
- Conversation history (last 10 turns for context)
- Cost accumulation (per-session tracking)
- Active utterances (for frame synchronization)
- Performance metrics (latency, success rate)

The system uses an async semaphore to respect ElevenLabs' concurrent request limits and implements exponential backoff for rate limit errors.

## Performance

| Metric | Value |
|--------|-------|
| Avg Latency | 2-4 seconds |
| P95 Latency | < 5 seconds |
| Max Sessions | 50+ |
| Success Rate | 98%+ |

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open a pull request

## License

MIT - see [LICENSE](LICENSE) file

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) - Async web framework
- [OpenAI GPT-4](https://openai.com/) - Language model
- [ElevenLabs](https://elevenlabs.io/) - Voice synthesis and transcription
- [Plotly](https://plotly.com/) - Data visualization

---

If you find this useful, star the repo!
