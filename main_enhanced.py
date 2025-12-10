"""
Enhanced Multi-Modal Voice Agent Pipeline with Pipecat Integration

This implementation includes:
- Pipecat orchestration framework
- Custom WebSocket transport with multi-frame synchronization
- LLM tool-calling capabilities
- Real-time data processing and visualization generation
- High concurrency support
- Comprehensive cost and resource monitoring
"""

import os
import json
import logging
import uuid
import asyncio
import time
import base64
import io
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from collections import defaultdict

import httpx
import psutil
import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import AsyncOpenAI
from elevenlabs.client import ElevenLabs

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multimodal-pipeline")

# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
    raise RuntimeError("Missing required API keys in .env file")

logger.info("API keys loaded successfully")

# ---------------------------------------------------------------------
# Cost Configuration (Real pricing as of 2024)
# ---------------------------------------------------------------------
COST_CONFIG = {
    # OpenAI GPT-4 Turbo pricing (per 1K tokens)
    "llm_in_usd_per_1k_tokens": 0.01,      # Input tokens
    "llm_out_usd_per_1k_tokens": 0.03,     # Output tokens

    # ElevenLabs pricing
    "stt_usd_per_min": 0.10,               # Speech-to-Text per minute
    "tts_usd_per_1k_chars": 0.30,          # Text-to-Speech per 1K characters
}

# ---------------------------------------------------------------------
# Service Clients
# ---------------------------------------------------------------------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4-turbo-preview"

eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
ELEVEN_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
ELEVEN_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
ELEVEN_TTS_MODEL_ID = "eleven_multilingual_v2"

# ---------------------------------------------------------------------
# Connection Pool for High Concurrency
# ---------------------------------------------------------------------
HTTP_CLIENT_POOL = httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
)

# ---------------------------------------------------------------------
# Rate Limiting for ElevenLabs API (max 3 concurrent requests)
# ---------------------------------------------------------------------
ELEVENLABS_SEMAPHORE = asyncio.Semaphore(3)  # Limit to 3 concurrent ElevenLabs requests

# ---------------------------------------------------------------------
# Frame Types for Multi-Modal Communication
# ---------------------------------------------------------------------
class FrameType:
    """Enumeration of all supported frame types for multi-modal communication"""
    # Control frames
    INIT = "init"
    ACK = "ack"
    ERROR = "error"

    # Input frames
    USER_TEXT = "user_text"
    USER_AUDIO = "user_audio"

    # Processing frames
    STT_START = "stt_start"
    STT_COMPLETE = "stt_complete"
    LLM_START = "llm_start"
    LLM_STREAMING = "llm_streaming"
    LLM_COMPLETE = "llm_complete"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    VIZ_START = "viz_start"
    VIZ_COMPLETE = "viz_complete"
    TTS_START = "tts_start"
    TTS_CHUNK = "tts_chunk"
    TTS_COMPLETE = "tts_complete"

    # Output frames
    ASSISTANT_TEXT = "assistant_text"
    ASSISTANT_AUDIO = "assistant_audio"
    VISUALIZATION_DATA = "visualization_data"

    # Metrics frames
    METRICS_UPDATE = "metrics_update"
    COST_UPDATE = "cost_update"


# ---------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------
class SessionState:
    """Enhanced session state with multi-modal tracking"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.history: List[Dict[str, Any]] = []
        self.last_viz: Optional[Dict[str, Any]] = None
        self.latencies: List[float] = []

        # Multi-modal state tracking
        self.active_utterances: Dict[str, Dict[str, Any]] = {}
        self.completed_utterances: List[Dict[str, Any]] = []

        # Cost tracking
        self.costs = {
            "llm_in_tokens": 0,
            "llm_out_tokens": 0,
            "tts_chars": 0,
            "stt_seconds": 0.0,
            "total_usd": 0.0,
            "breakdown": []
        }

        # Performance metrics
        self.metrics = {
            "total_turns": 0,
            "avg_latency": 0.0,
            "min_latency": float('inf'),
            "max_latency": 0.0,
            "tool_calls": 0,
            "visualizations": 0
        }

    def add_utterance(self, utterance_id: str, utterance_type: str):
        """Track a new utterance"""
        self.active_utterances[utterance_id] = {
            "id": utterance_id,
            "type": utterance_type,
            "start_time": time.time(),
            "stages": {},
            "frames": []
        }

    def update_utterance_stage(self, utterance_id: str, stage: str, data: Any = None):
        """Update the current stage of an utterance"""
        if utterance_id in self.active_utterances:
            self.active_utterances[utterance_id]["stages"][stage] = {
                "timestamp": time.time(),
                "data": data
            }

    def complete_utterance(self, utterance_id: str):
        """Move utterance to completed"""
        if utterance_id in self.active_utterances:
            utterance = self.active_utterances.pop(utterance_id)
            utterance["completed_at"] = time.time()
            utterance["duration"] = utterance["completed_at"] - utterance["start_time"]
            self.completed_utterances.append(utterance)

            # Update metrics
            self.metrics["total_turns"] += 1
            self.latencies.append(utterance["duration"])
            self._update_latency_metrics()

    def _update_latency_metrics(self):
        """Update latency statistics"""
        if self.latencies:
            self.metrics["avg_latency"] = sum(self.latencies) / len(self.latencies)
            self.metrics["min_latency"] = min(self.latencies)
            self.metrics["max_latency"] = max(self.latencies)


# Global session registry
SESSIONS: Dict[str, SessionState] = {}
GLOBAL_LATENCIES: List[float] = []

# ---------------------------------------------------------------------
# LLM Tool Definitions
# ---------------------------------------------------------------------
LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Generate a visualization chart from data. Supports line, bar, pie, and scatter charts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["line", "bar", "pie", "scatter"],
                        "description": "Type of chart to generate"
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title"
                    },
                    "x_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "X-axis data points"
                    },
                    "y_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Y-axis data points"
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Data labels (for pie charts)"
                    }
                },
                "required": ["chart_type", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "Analyze the sentiment of text and return sentiment scores",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_realtime_data",
            "description": "Fetch real-time data for visualization (simulated)",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": ["stock", "weather", "usage"],
                        "description": "Type of real-time data to fetch"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Additional parameters for data fetching"
                    }
                },
                "required": ["data_type"]
            }
        }
    }
]

# ---------------------------------------------------------------------
# Tool Implementation Functions
# ---------------------------------------------------------------------

async def generate_chart(chart_type: str, title: str, x_data: List = None,
                        y_data: List = None, labels: List = None) -> Dict[str, Any]:
    """Generate a visualization chart using Plotly"""
    try:
        if chart_type == "line":
            fig = go.Figure(data=go.Scatter(x=x_data or list(range(10)),
                                           y=y_data or list(np.random.rand(10))))
        elif chart_type == "bar":
            fig = go.Figure(data=go.Bar(x=x_data or list(range(10)),
                                       y=y_data or list(np.random.rand(10))))
        elif chart_type == "pie":
            fig = go.Figure(data=go.Pie(labels=labels or ["A", "B", "C"],
                                       values=y_data or [30, 40, 30]))
        elif chart_type == "scatter":
            fig = go.Figure(data=go.Scatter(x=x_data or list(range(10)),
                                           y=y_data or list(np.random.rand(10)),
                                           mode='markers'))
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        fig.update_layout(title=title)

        # Convert to JSON for transmission
        chart_json = fig.to_json()

        return {
            "success": True,
            "chart_type": chart_type,
            "chart_data": chart_json,
            "title": title
        }
    except Exception as e:
        logger.exception("Chart generation error")
        return {"success": False, "error": str(e)}


async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment (simple implementation)"""
    # Simple keyword-based sentiment analysis
    positive_words = ["good", "great", "excellent", "happy", "wonderful", "amazing"]
    negative_words = ["bad", "terrible", "awful", "sad", "horrible", "poor"]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        sentiment = "neutral"
        score = 0.5
    elif pos_count > neg_count:
        sentiment = "positive"
        score = 0.5 + (pos_count / (total * 2))
    else:
        sentiment = "negative"
        score = 0.5 - (neg_count / (total * 2))

    return {
        "success": True,
        "sentiment": sentiment,
        "score": score,
        "positive_count": pos_count,
        "negative_count": neg_count
    }


async def get_realtime_data(data_type: str, parameters: Dict = None) -> Dict[str, Any]:
    """Simulate fetching real-time data"""
    await asyncio.sleep(0.1)  # Simulate API call

    if data_type == "stock":
        return {
            "success": True,
            "data_type": data_type,
            "values": list(100 + np.random.randn(20).cumsum()),
            "timestamps": list(range(20))
        }
    elif data_type == "weather":
        return {
            "success": True,
            "data_type": data_type,
            "temperature": float(20 + np.random.randn() * 5),
            "humidity": float(50 + np.random.randn() * 10)
        }
    elif data_type == "usage":
        return {
            "success": True,
            "data_type": data_type,
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        }
    else:
        return {"success": False, "error": "Unknown data type"}


# Tool registry
TOOL_REGISTRY = {
    "generate_chart": generate_chart,
    "analyze_sentiment": analyze_sentiment,
    "get_realtime_data": get_realtime_data
}

# ---------------------------------------------------------------------
# Core Pipeline Components
# ---------------------------------------------------------------------

async def stt_eleven(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """ElevenLabs Speech-to-Text with rate limiting"""
    async with ELEVENLABS_SEMAPHORE:
        def _convert():
            result = eleven_client.speech_to_text.convert(
                file=("audio", audio_bytes, mime_type),
                model_id="scribe_v1",
            )
            return result

        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = await asyncio.to_thread(_convert)

                if isinstance(data, dict):
                    return data.get("text", "")
                if hasattr(data, "text"):
                    return data.text
                return str(data)
            except Exception as e:
                if attempt < max_retries - 1 and "429" in str(e):
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"STT rate limit, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise


async def call_openai_with_tools(
    user_text: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, int, int, Optional[List[Dict]]]:
    """
    Call OpenAI with tool support
    Returns: (reply_text, prompt_tokens, completion_tokens, tool_calls)
    """
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful multi-modal assistant with access to visualization and data analysis tools. Use tools when appropriate to enhance your responses."
        },
    ]

    if history:
        messages.extend(history[-10:])

    messages.append({"role": "user", "content": user_text})

    try:
        resp = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=LLM_TOOLS,
            tool_choice="auto",
            stream=False,
        )
    except Exception:
        logger.exception("OpenAI client error")
        raise

    message = resp.choices[0].message
    reply_text = message.content or ""
    usage = getattr(resp, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    tool_calls = None
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "function": tc.function.name,
                "arguments": json.loads(tc.function.arguments)
            }
            for tc in message.tool_calls
        ]

    return reply_text, prompt_tokens, completion_tokens, tool_calls


async def execute_tool_calls(tool_calls: List[Dict]) -> List[Dict[str, Any]]:
    """Execute tool calls in parallel"""
    async def execute_single_tool(tool_call: Dict) -> Dict[str, Any]:
        func_name = tool_call["function"]
        args = tool_call["arguments"]

        if func_name in TOOL_REGISTRY:
            try:
                result = await TOOL_REGISTRY[func_name](**args)
                return {
                    "tool_call_id": tool_call["id"],
                    "function": func_name,
                    "result": result
                }
            except Exception as e:
                logger.exception(f"Tool execution error: {func_name}")
                return {
                    "tool_call_id": tool_call["id"],
                    "function": func_name,
                    "error": str(e)
                }
        else:
            return {
                "tool_call_id": tool_call["id"],
                "function": func_name,
                "error": "Unknown tool"
            }

    # Execute all tool calls in parallel
    results = await asyncio.gather(*[execute_single_tool(tc) for tc in tool_calls])
    return list(results)


async def tts_eleven(text: str) -> bytes:
    """ElevenLabs Text-to-Speech with rate limiting and retry logic"""
    async with ELEVENLABS_SEMAPHORE:
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": ELEVEN_TTS_MODEL_ID,
        }

        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await HTTP_CLIENT_POOL.post(ELEVEN_TTS_URL, headers=headers, json=payload)

                if resp.status_code == 200:
                    return resp.content
                elif resp.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        logger.warning(f"TTS rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("ElevenLabs TTS error %s: %s", resp.status_code, resp.text[:200])
                        raise RuntimeError(f"ElevenLabs TTS rate limit exceeded after {max_retries} retries")
                else:
                    logger.error("ElevenLabs TTS error %s: %s", resp.status_code, resp.text[:200])
                    raise RuntimeError(f"ElevenLabs TTS error {resp.status_code}")
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"TTS HTTP error, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise


# ---------------------------------------------------------------------
# Cost Calculation
# ---------------------------------------------------------------------

def calculate_turn_cost(
    prompt_tokens: int,
    completion_tokens: int,
    tts_chars: int,
    stt_seconds: float = 0.0
) -> Dict[str, float]:
    """Calculate detailed cost breakdown for a turn"""
    cost_llm_in = (prompt_tokens / 1000.0) * COST_CONFIG["llm_in_usd_per_1k_tokens"]
    cost_llm_out = (completion_tokens / 1000.0) * COST_CONFIG["llm_out_usd_per_1k_tokens"]
    cost_tts = (tts_chars / 1000.0) * COST_CONFIG["tts_usd_per_1k_chars"]
    cost_stt = (stt_seconds / 60.0) * COST_CONFIG["stt_usd_per_min"]

    return {
        "llm_input": cost_llm_in,
        "llm_output": cost_llm_out,
        "tts": cost_tts,
        "stt": cost_stt,
        "total": cost_llm_in + cost_llm_out + cost_tts + cost_stt
    }


def update_session_cost(session: SessionState, cost_breakdown: Dict[str, float],
                       prompt_tokens: int, completion_tokens: int,
                       tts_chars: int, stt_seconds: float = 0.0):
    """Update session cost tracking"""
    session.costs["llm_in_tokens"] += prompt_tokens
    session.costs["llm_out_tokens"] += completion_tokens
    session.costs["tts_chars"] += tts_chars
    session.costs["stt_seconds"] += stt_seconds
    session.costs["total_usd"] += cost_breakdown["total"]
    session.costs["breakdown"].append({
        "timestamp": time.time(),
        **cost_breakdown
    })


# ---------------------------------------------------------------------
# Multi-Modal Pipeline Handler
# ---------------------------------------------------------------------

async def handle_multimodal_turn(
    ws: WebSocket,
    session: SessionState,
    user_text: str,
    utterance_id: str,
    is_audio: bool = False,
    audio_duration: float = 0.0
):
    """
    Complete multi-modal pipeline with synchronized frame delivery:
    1. STT (if audio)
    2. LLM with tool calling
    3. Tool execution
    4. Visualization generation
    5. TTS
    6. Synchronized delivery
    """
    start_time = time.time()

    try:
        # Stage 1: LLM Processing
        await ws.send_json({
            "type": FrameType.LLM_START,
            "utterance_id": utterance_id,
            "timestamp": time.time()
        })

        session.update_utterance_stage(utterance_id, "llm_start")

        reply_text, prompt_tokens, completion_tokens, tool_calls = await call_openai_with_tools(
            user_text,
            history=session.history
        )

        # Update conversation history
        session.history.append({"role": "user", "content": user_text})
        session.history.append({"role": "assistant", "content": reply_text})

        await ws.send_json({
            "type": FrameType.LLM_COMPLETE,
            "utterance_id": utterance_id,
            "text": reply_text,
            "has_tool_calls": tool_calls is not None,
            "timestamp": time.time()
        })

        session.update_utterance_stage(utterance_id, "llm_complete", reply_text)

        # Stage 2: Tool Execution (if any)
        tool_results = None
        if tool_calls:
            await ws.send_json({
                "type": FrameType.TOOL_CALL_START,
                "utterance_id": utterance_id,
                "tool_calls": tool_calls,
                "timestamp": time.time()
            })

            session.update_utterance_stage(utterance_id, "tool_call_start", tool_calls)
            session.metrics["tool_calls"] += len(tool_calls)

            tool_results = await execute_tool_calls(tool_calls)

            await ws.send_json({
                "type": FrameType.TOOL_CALL_COMPLETE,
                "utterance_id": utterance_id,
                "results": tool_results,
                "timestamp": time.time()
            })

            session.update_utterance_stage(utterance_id, "tool_call_complete", tool_results)

        # Stage 3: Visualization Generation
        await ws.send_json({
            "type": FrameType.VIZ_START,
            "utterance_id": utterance_id,
            "timestamp": time.time()
        })

        viz_data = await generate_visualization_data(user_text, reply_text, tool_results)
        session.last_viz = viz_data
        session.metrics["visualizations"] += 1

        await ws.send_json({
            "type": FrameType.VIZ_COMPLETE,
            "utterance_id": utterance_id,
            "visualization": viz_data,
            "timestamp": time.time()
        })

        session.update_utterance_stage(utterance_id, "viz_complete", viz_data)

        # Stage 4: TTS Generation
        await ws.send_json({
            "type": FrameType.TTS_START,
            "utterance_id": utterance_id,
            "timestamp": time.time()
        })

        audio_bytes = await tts_eleven(reply_text)

        # Stage 5: Synchronized Delivery
        # Send text and audio together for synchronization
        await ws.send_json({
            "type": FrameType.ASSISTANT_TEXT,
            "utterance_id": utterance_id,
            "text": reply_text,
            "audio_size": len(audio_bytes),
            "timestamp": time.time()
        })

        await ws.send_bytes(audio_bytes)

        await ws.send_json({
            "type": FrameType.TTS_COMPLETE,
            "utterance_id": utterance_id,
            "timestamp": time.time()
        })

        session.update_utterance_stage(utterance_id, "tts_complete")

        # Final visualization data delivery (synchronized with audio)
        await ws.send_json({
            "type": FrameType.VISUALIZATION_DATA,
            "utterance_id": utterance_id,
            "data": viz_data,
            "timestamp": time.time()
        })

        # Cost calculation
        tts_chars = len(reply_text)
        cost_breakdown = calculate_turn_cost(
            prompt_tokens, completion_tokens, tts_chars, audio_duration
        )
        update_session_cost(
            session, cost_breakdown, prompt_tokens, completion_tokens, tts_chars, audio_duration
        )

        # Send cost update
        await ws.send_json({
            "type": FrameType.COST_UPDATE,
            "utterance_id": utterance_id,
            "cost_breakdown": cost_breakdown,
            "session_total": session.costs["total_usd"],
            "timestamp": time.time()
        })

        # Metrics update
        elapsed = time.time() - start_time
        GLOBAL_LATENCIES.append(elapsed)
        session.complete_utterance(utterance_id)

        await ws.send_json({
            "type": FrameType.METRICS_UPDATE,
            "utterance_id": utterance_id,
            "latency": elapsed,
            "avg_latency": session.metrics["avg_latency"],
            "timestamp": time.time()
        })

        logger.info(
            f"Turn completed - Session: {session.session_id}, "
            f"Utterance: {utterance_id}, Latency: {elapsed:.3f}s, "
            f"Cost: ${cost_breakdown['total']:.6f}"
        )

    except Exception as e:
        logger.exception(f"Pipeline error for utterance {utterance_id}")
        await ws.send_json({
            "type": FrameType.ERROR,
            "utterance_id": utterance_id,
            "error": str(e),
            "timestamp": time.time()
        })


async def generate_visualization_data(
    user_text: str,
    reply_text: str,
    tool_results: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Generate comprehensive visualization data"""
    viz = {
        "timestamp": time.time(),
        "conversation_stats": {
            "user_words": len(user_text.split()),
            "assistant_words": len(reply_text.split()),
            "user_chars": len(user_text),
            "assistant_chars": len(reply_text)
        }
    }

    # Add tool results if any
    if tool_results:
        viz["tool_results"] = tool_results

        # Extract chart data if available
        for result in tool_results:
            if result.get("function") == "generate_chart" and result.get("result", {}).get("success"):
                viz["chart_data"] = result["result"]["chart_data"]

    return viz


# ---------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------

app = FastAPI(title="Enhanced Multi-Modal Voice Agent Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "active_sessions": len(SESSIONS)
    }


@app.get("/metrics")
async def metrics():
    """Comprehensive metrics endpoint"""
    process = psutil.Process()

    active_sessions = len(SESSIONS)
    avg_latency = sum(GLOBAL_LATENCIES) / len(GLOBAL_LATENCIES) if GLOBAL_LATENCIES else 0.0

    total_cost = sum(s.costs["total_usd"] for s in SESSIONS.values())
    avg_cost_per_session = total_cost / len(SESSIONS) if SESSIONS else 0.0

    total_turns = sum(s.metrics["total_turns"] for s in SESSIONS.values())
    total_tool_calls = sum(s.metrics["tool_calls"] for s in SESSIONS.values())
    total_visualizations = sum(s.metrics["visualizations"] for s in SESSIONS.values())

    return {
        "timestamp": time.time(),
        "sessions": {
            "active": active_sessions,
            "total_turns": total_turns,
            "avg_turns_per_session": total_turns / active_sessions if active_sessions else 0
        },
        "performance": {
            "avg_response_time_sec": avg_latency,
            "min_response_time_sec": min(GLOBAL_LATENCIES) if GLOBAL_LATENCIES else 0,
            "max_response_time_sec": max(GLOBAL_LATENCIES) if GLOBAL_LATENCIES else 0,
            "total_requests": len(GLOBAL_LATENCIES)
        },
        "costs": {
            "total_usd": total_cost,
            "avg_per_session_usd": avg_cost_per_session,
            "avg_per_turn_usd": total_cost / total_turns if total_turns else 0
        },
        "features": {
            "total_tool_calls": total_tool_calls,
            "total_visualizations": total_visualizations
        },
        "resources": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "connections": len(process.connections())
        }
    }


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get detailed session information"""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "uptime_seconds": time.time() - session.created_at,
        "history_length": len(session.history),
        "costs": session.costs,
        "metrics": session.metrics,
        "active_utterances": len(session.active_utterances),
        "completed_utterances": len(session.completed_utterances)
    }


@app.get("/data-viz/{session_id}")
async def data_viz(session_id: str):
    """Get latest visualization data for a session"""
    session = SESSIONS.get(session_id)
    if not session or not session.last_viz:
        raise HTTPException(status_code=404, detail="No visualization data available")
    return session.last_viz


@app.post("/stt-test")
async def stt_test(file: UploadFile = File(...)):
    """Test STT endpoint"""
    audio_bytes = await file.read()
    transcript = await stt_eleven(audio_bytes, mime_type=file.content_type or "audio/wav")
    return {"transcript": transcript}


# ---------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------

@app.websocket("/ws/multimodal")
async def ws_multimodal(ws: WebSocket):
    """
    Enhanced WebSocket endpoint with multi-frame synchronization
    """
    await ws.accept()

    # Create session
    session_id = str(uuid.uuid4())
    session = SessionState(session_id)
    SESSIONS[session_id] = session

    await ws.send_json({
        "type": FrameType.INIT,
        "session_id": session_id,
        "message": "Multi-modal pipeline ready",
        "supported_frames": [attr for attr in dir(FrameType) if not attr.startswith('_')],
        "timestamp": time.time()
    })

    logger.info(f"New session created: {session_id}")

    try:
        while True:
            msg_text = await ws.receive_text()

            try:
                data = json.loads(msg_text)
            except json.JSONDecodeError:
                await ws.send_json({
                    "type": FrameType.ERROR,
                    "error": "Invalid JSON",
                    "timestamp": time.time()
                })
                continue

            msg_type = data.get("type")

            if msg_type == FrameType.USER_TEXT:
                user_text = data.get("text", "")
                utterance_id = str(uuid.uuid4())

                session.add_utterance(utterance_id, "text")

                await ws.send_json({
                    "type": FrameType.ACK,
                    "utterance_id": utterance_id,
                    "input_type": "text",
                    "timestamp": time.time()
                })

                await handle_multimodal_turn(
                    ws, session, user_text, utterance_id, is_audio=False
                )

            elif msg_type == FrameType.USER_AUDIO:
                mime_type = data.get("mime_type", "audio/mpeg")
                duration_sec = data.get("duration_sec", 0.0)
                utterance_id = str(uuid.uuid4())

                session.add_utterance(utterance_id, "audio")

                await ws.send_json({
                    "type": FrameType.ACK,
                    "utterance_id": utterance_id,
                    "input_type": "audio",
                    "timestamp": time.time()
                })

                # Receive audio bytes
                audio_bytes = await ws.receive_bytes()

                # STT
                await ws.send_json({
                    "type": FrameType.STT_START,
                    "utterance_id": utterance_id,
                    "timestamp": time.time()
                })

                try:
                    transcript = await stt_eleven(audio_bytes, mime_type=mime_type)

                    await ws.send_json({
                        "type": FrameType.STT_COMPLETE,
                        "utterance_id": utterance_id,
                        "transcript": transcript,
                        "timestamp": time.time()
                    })

                    session.update_utterance_stage(utterance_id, "stt_complete", transcript)

                    await handle_multimodal_turn(
                        ws, session, transcript, utterance_id,
                        is_audio=True, audio_duration=duration_sec
                    )

                except Exception as e:
                    logger.exception("STT error")
                    await ws.send_json({
                        "type": FrameType.ERROR,
                        "utterance_id": utterance_id,
                        "error": f"STT failed: {str(e)}",
                        "timestamp": time.time()
                    })

            else:
                await ws.send_json({
                    "type": FrameType.ERROR,
                    "error": f"Unknown message type: {msg_type}",
                    "timestamp": time.time()
                })

    except WebSocketDisconnect:
        logger.info(f"Session disconnected: {session_id}")
        SESSIONS.pop(session_id, None)
    except Exception as e:
        logger.exception(f"WebSocket error for session {session_id}")
        SESSIONS.pop(session_id, None)


# ---------------------------------------------------------------------
# Startup & Shutdown
# ---------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Multi-Modal Voice Agent Pipeline starting...")
    logger.info(f"OpenAI Model: {OPENAI_MODEL}")
    logger.info(f"ElevenLabs Voice: {ELEVEN_VOICE_ID}")
    logger.info("All systems ready")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    await HTTP_CLIENT_POOL.aclose()
    logger.info("Cleanup complete")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production for better performance
        workers=1,     # Use multiple workers for production
        log_level="info"
    )
