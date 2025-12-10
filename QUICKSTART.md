# Quick Start Guide - Multi-Modal Voice Agent Pipeline

## 🚀 Get Running in 5 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
cd /home/para/work/job-asign/adiiva/multimodal-demo
pip install -r requirements.txt
```

### Step 2: Configure API Keys (1 minute)

Create a `.env` file in the project directory:

```bash
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
EOF
```

**Get your API keys:**
- OpenAI: https://platform.openai.com/api-keys
- ElevenLabs: https://elevenlabs.io/app/settings/api-keys

### Step 3: Start the Server (30 seconds)

```bash
python main_enhanced.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Open the Client (30 seconds)

Open `client_enhanced.html` in your web browser:

```bash
# On Mac
open client_enhanced.html

# On Linux
xdg-open client_enhanced.html

# On Windows
start client_enhanced.html
```

### Step 5: Test It! (2 minutes)

1. Click **"Connect"** button
2. Type a message: "Create a bar chart with values 10, 20, 30, 40"
3. Click **"Send Text"**
4. Watch the magic happen! 🎉

---

## 🎯 Quick Test Commands

### Test the Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "timestamp": 1234567890.123, "active_sessions": 0}
```

### Test Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

### Test STT Endpoint (if you have an audio file)

```bash
curl -X POST -F "file=@test_audio.wav" http://localhost:8000/stt-test
```

---

## 🧪 Run Performance Benchmark

```bash
python benchmark.py --max-concurrency 10
```

This will:
- Test the system with 1, 5, 10 concurrent connections
- Generate a detailed performance report
- Save results to `benchmark_report.json`

---

## 🎨 Try Different Test Messages

In the client, click these quick test buttons:

1. **Greeting**: "Hello, how are you?"
2. **Chart**: "Create a bar chart with values 10, 20, 30, 40"
3. **Sentiment**: "Analyze sentiment: I love this product!"
4. **Usage Data**: "Get real-time usage data"

---

## 📊 Generate Cost Analysis Report

Create a Python script to generate cost report:

```python
from cost_analysis import CostAnalyzer, ResourceMonitor, generate_combined_report, save_report_to_file
from main_enhanced import SESSIONS, COST_CONFIG

cost_analyzer = CostAnalyzer(COST_CONFIG)
resource_monitor = ResourceMonitor()

# After running some sessions...
report = generate_combined_report(cost_analyzer, resource_monitor, SESSIONS)
filename = save_report_to_file(report)
print(f"Report saved to: {filename}")
```

---

## 🔧 Troubleshooting

### "Connection Refused" Error

**Problem**: Can't connect to the server

**Solution**:
```bash
# Check if server is running
ps aux | grep main_enhanced

# Check if port 8000 is in use
lsof -i :8000

# Restart the server
python main_enhanced.py
```

### "Invalid API Key" Error

**Problem**: API keys not configured correctly

**Solution**:
1. Check `.env` file exists in the same directory as `main_enhanced.py`
2. Verify API keys are correct (no quotes, no spaces)
3. Restart the server after updating `.env`

### "Module Not Found" Error

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### WebSocket Connection Issues

**Problem**: WebSocket fails to connect

**Solution**:
1. Check browser console for errors (F12)
2. Verify WebSocket URL is correct: `ws://localhost:8000/ws/multimodal`
3. Check firewall settings
4. Try a different browser

---

## 🎤 Testing Audio Features

### Requirements

- Modern browser (Chrome, Firefox, Edge)
- Microphone access permission
- WebM audio codec support

### How to Test

1. Click **"Connect"**
2. Click **"🎤 Record Audio"**
3. Allow microphone access when prompted
4. Speak your message
5. Click **"⏹️ Stop Recording"**
6. Wait for response

---

## 📈 Monitoring Performance

### Real-Time Metrics

Watch the metrics while using the system:

```bash
# In another terminal
watch -n 1 'curl -s http://localhost:8000/metrics | json_pp'
```

### Resource Usage

```bash
# Monitor CPU and memory
top -p $(pgrep -f main_enhanced)
```

---

## 🔄 Switching Between Implementations

### Use Enhanced Version (Recommended)

```bash
python main_enhanced.py
```

Features:
- ✅ Tool calling
- ✅ Advanced synchronization
- ✅ Cost tracking
- ✅ Resource monitoring
- ✅ 19+ frame types

### Use Original Version

```bash
python main.py
```

Features:
- ✅ Basic pipeline
- ✅ Text and audio input
- ✅ Simple visualization
- ❌ No tool calling
- ❌ Limited metrics

---

## 🌐 Production Deployment

### Quick Production Setup

```bash
# Install gunicorn or use uvicorn with workers
pip install gunicorn

# Run with 4 workers
gunicorn main_enhanced:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables for Production

```bash
# .env
OPENAI_API_KEY=your_key
ELEVENLABS_API_KEY=your_key

# Optional: override model selection
OPENAI_MODEL=gpt-3.5-turbo  # Faster and cheaper
```

---

## 📚 Next Steps

After getting it running:

1. **Read the full documentation**: [README.md](README.md)
2. **Review implementation details**: [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)
3. **Run benchmarks**: `python benchmark.py`
4. **Customize tools**: Edit `LLM_TOOLS` in `main_enhanced.py`
5. **Add monitoring**: Integrate Prometheus/Grafana

---

## 💡 Pro Tips

### Reduce Costs

1. Switch to GPT-3.5-Turbo:
   ```python
   OPENAI_MODEL = "gpt-3.5-turbo"  # 90% cheaper
   ```

2. Limit response length:
   ```python
   resp = await openai_client.chat.completions.create(
       model=OPENAI_MODEL,
       messages=messages,
       max_tokens=150  # Add this
   )
   ```

### Improve Latency

1. Use connection pooling (already enabled)
2. Reduce conversation history (change from 10 to 5 turns)
3. Use streaming responses (requires implementation)
4. Deploy closer to API servers

### Handle More Concurrency

1. Use multiple workers:
   ```bash
   uvicorn main_enhanced:app --workers 4
   ```

2. Deploy multiple instances behind load balancer
3. Use Redis for session storage
4. Implement connection limits per client

---

## 🆘 Getting Help

### Check Logs

```bash
# Server logs
python main_enhanced.py 2>&1 | tee server.log

# Tail logs in real-time
tail -f server.log
```

### Common Issues

**Issue**: Slow responses
- **Check**: Network latency to OpenAI/ElevenLabs
- **Solution**: Use faster model or reduce history

**Issue**: High costs
- **Check**: `/metrics` endpoint for cost breakdown
- **Solution**: Implement response length limits

**Issue**: Memory leaks
- **Check**: `/metrics` endpoint for memory growth
- **Solution**: Restart server, review session cleanup

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Server starts without errors
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] Client connects via WebSocket
- [ ] Text messages work end-to-end
- [ ] Audio recording works (if needed)
- [ ] Visualizations render correctly
- [ ] Cost tracking shows non-zero values
- [ ] Metrics endpoint returns data
- [ ] Benchmark script runs successfully

---

## 🎉 You're Ready!

Your Multi-Modal Voice Agent Pipeline is now running!

**Access Points**:
- Web Client: `client_enhanced.html`
- API: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws/multimodal`
- Metrics: `http://localhost:8000/metrics`

**Have fun building amazing voice applications!** 🚀
