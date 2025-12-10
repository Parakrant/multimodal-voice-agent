"""
Comprehensive Cost and Resource Analysis Module

Provides detailed cost analysis reports for the Multi-Modal Voice Agent Pipeline
"""

import json
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta
import psutil


class CostAnalyzer:
    """Comprehensive cost analysis for multi-modal pipeline"""

    def __init__(self, cost_config: Dict[str, float]):
        self.cost_config = cost_config
        self.session_data: Dict[str, Dict] = {}

    def track_session(self, session_id: str, session_state: Any):
        """Track a session's cost data"""
        self.session_data[session_id] = {
            "costs": session_state.costs,
            "metrics": session_state.metrics,
            "created_at": session_state.created_at,
            "history_length": len(session_state.history)
        }

    def generate_cost_report(self, sessions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive cost analysis report

        Returns detailed breakdown of:
        - Per-session costs
        - Per-minute operational costs
        - Component-wise cost distribution
        - Cost projections
        """
        if not sessions:
            return {
                "error": "No active sessions to analyze",
                "timestamp": time.time()
            }

        # Aggregate data
        total_sessions = len(sessions)
        total_llm_in_tokens = 0
        total_llm_out_tokens = 0
        total_tts_chars = 0
        total_stt_seconds = 0.0
        total_cost_usd = 0.0
        total_turns = 0
        total_duration_seconds = 0.0

        session_costs = []

        for session_id, session in sessions.items():
            costs = session.costs
            metrics = session.metrics
            duration = time.time() - session.created_at

            total_llm_in_tokens += costs["llm_in_tokens"]
            total_llm_out_tokens += costs["llm_out_tokens"]
            total_tts_chars += costs["tts_chars"]
            total_stt_seconds += costs["stt_seconds"]
            total_cost_usd += costs["total_usd"]
            total_turns += metrics["total_turns"]
            total_duration_seconds += duration

            session_costs.append({
                "session_id": session_id,
                "total_cost_usd": costs["total_usd"],
                "turns": metrics["total_turns"],
                "duration_seconds": duration,
                "cost_per_turn": costs["total_usd"] / metrics["total_turns"] if metrics["total_turns"] > 0 else 0,
                "cost_per_minute": (costs["total_usd"] / duration) * 60 if duration > 0 else 0
            })

        # Calculate averages
        avg_cost_per_session = total_cost_usd / total_sessions
        avg_turns_per_session = total_turns / total_sessions
        avg_duration_per_session = total_duration_seconds / total_sessions
        avg_cost_per_turn = total_cost_usd / total_turns if total_turns > 0 else 0

        # Per-minute cost calculation
        total_minutes = total_duration_seconds / 60
        cost_per_minute = total_cost_usd / total_minutes if total_minutes > 0 else 0

        # Component-wise breakdown
        llm_input_cost = (total_llm_in_tokens / 1000) * self.cost_config["llm_in_usd_per_1k_tokens"]
        llm_output_cost = (total_llm_out_tokens / 1000) * self.cost_config["llm_out_usd_per_1k_tokens"]
        tts_cost = (total_tts_chars / 1000) * self.cost_config["tts_usd_per_1k_chars"]
        stt_cost = (total_stt_seconds / 60) * self.cost_config["stt_usd_per_min"]

        component_breakdown = {
            "llm_input": {
                "cost_usd": llm_input_cost,
                "percentage": (llm_input_cost / total_cost_usd * 100) if total_cost_usd > 0 else 0,
                "tokens": total_llm_in_tokens,
                "rate": f"${self.cost_config['llm_in_usd_per_1k_tokens']}/1K tokens"
            },
            "llm_output": {
                "cost_usd": llm_output_cost,
                "percentage": (llm_output_cost / total_cost_usd * 100) if total_cost_usd > 0 else 0,
                "tokens": total_llm_out_tokens,
                "rate": f"${self.cost_config['llm_out_usd_per_1k_tokens']}/1K tokens"
            },
            "tts": {
                "cost_usd": tts_cost,
                "percentage": (tts_cost / total_cost_usd * 100) if total_cost_usd > 0 else 0,
                "characters": total_tts_chars,
                "rate": f"${self.cost_config['tts_usd_per_1k_chars']}/1K chars"
            },
            "stt": {
                "cost_usd": stt_cost,
                "percentage": (stt_cost / total_cost_usd * 100) if total_cost_usd > 0 else 0,
                "seconds": total_stt_seconds,
                "rate": f"${self.cost_config['stt_usd_per_min']}/minute"
            }
        }

        # Projections
        daily_projection = cost_per_minute * 60 * 24 if cost_per_minute > 0 else 0
        monthly_projection = daily_projection * 30
        yearly_projection = daily_projection * 365

        # Cost efficiency metrics
        tokens_per_dollar = (total_llm_in_tokens + total_llm_out_tokens) / total_cost_usd if total_cost_usd > 0 else 0
        chars_per_dollar = total_tts_chars / total_cost_usd if total_cost_usd > 0 else 0

        return {
            "timestamp": time.time(),
            "report_period": {
                "total_sessions": total_sessions,
                "total_duration_seconds": total_duration_seconds,
                "total_duration_minutes": total_minutes,
                "total_turns": total_turns
            },
            "cost_summary": {
                "total_usd": round(total_cost_usd, 6),
                "avg_per_session_usd": round(avg_cost_per_session, 6),
                "avg_per_turn_usd": round(avg_cost_per_turn, 6),
                "cost_per_minute_usd": round(cost_per_minute, 6)
            },
            "component_breakdown": component_breakdown,
            "session_details": sorted(session_costs, key=lambda x: x["total_cost_usd"], reverse=True),
            "projections": {
                "note": "Based on current usage rate",
                "daily_usd": round(daily_projection, 2),
                "monthly_usd": round(monthly_projection, 2),
                "yearly_usd": round(yearly_projection, 2)
            },
            "efficiency_metrics": {
                "tokens_per_dollar": round(tokens_per_dollar, 2),
                "chars_per_dollar": round(chars_per_dollar, 2),
                "turns_per_dollar": round(1 / avg_cost_per_turn, 2) if avg_cost_per_turn > 0 else 0
            },
            "recommendations": self._generate_recommendations(component_breakdown, avg_cost_per_turn)
        }

    def _generate_recommendations(self, component_breakdown: Dict, avg_cost_per_turn: float) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []

        # Analyze component costs
        llm_output_pct = component_breakdown["llm_output"]["percentage"]
        tts_pct = component_breakdown["tts"]["percentage"]
        llm_input_pct = component_breakdown["llm_input"]["percentage"]

        if llm_output_pct > 50:
            recommendations.append(
                f"LLM output tokens account for {llm_output_pct:.1f}% of costs. "
                "Consider using shorter responses or streaming for long outputs."
            )

        if tts_pct > 40:
            recommendations.append(
                f"TTS accounts for {tts_pct:.1f}% of costs. "
                "Consider implementing response length limits or caching common phrases."
            )

        if llm_input_pct > 30:
            recommendations.append(
                f"LLM input tokens account for {llm_input_pct:.1f}% of costs. "
                "Consider limiting conversation history or implementing summarization."
            )

        if avg_cost_per_turn > 0.10:
            recommendations.append(
                f"Average cost per turn (${avg_cost_per_turn:.4f}) is high. "
                "Consider optimizing prompt length and response verbosity."
            )

        if not recommendations:
            recommendations.append("Cost distribution is well-balanced. Continue monitoring.")

        return recommendations


class ResourceMonitor:
    """Real-time resource utilization monitoring"""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss
        self.baseline_time = time.time()
        self.samples: List[Dict[str, Any]] = []

    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture current resource usage snapshot"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_percent = psutil.virtual_memory().percent

        snapshot = {
            "timestamp": time.time(),
            "cpu": {
                "process_percent": cpu_percent,
                "system_percent": psutil.cpu_percent(interval=0.1),
                "cores": psutil.cpu_count()
            },
            "memory": {
                "process_mb": memory_info.rss / 1024 / 1024,
                "process_percent": (memory_info.rss / psutil.virtual_memory().total) * 100,
                "system_percent": memory_percent,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            },
            "threads": self.process.num_threads(),
            "connections": len(self.process.connections()),
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict()
        }

        self.samples.append(snapshot)
        return snapshot

    def generate_resource_report(self, active_sessions: int) -> Dict[str, Any]:
        """Generate comprehensive resource utilization report"""
        current_snapshot = self.capture_snapshot()
        uptime = time.time() - self.baseline_time

        # Calculate averages from samples
        if self.samples:
            avg_cpu = sum(s["cpu"]["process_percent"] for s in self.samples) / len(self.samples)
            avg_memory_mb = sum(s["memory"]["process_mb"] for s in self.samples) / len(self.samples)
            max_memory_mb = max(s["memory"]["process_mb"] for s in self.samples)
            max_threads = max(s["threads"] for s in self.samples)
            max_connections = max(s["connections"] for s in self.samples)
        else:
            avg_cpu = current_snapshot["cpu"]["process_percent"]
            avg_memory_mb = current_snapshot["memory"]["process_mb"]
            max_memory_mb = avg_memory_mb
            max_threads = current_snapshot["threads"]
            max_connections = current_snapshot["connections"]

        # Memory growth rate
        memory_growth_mb = (current_snapshot["memory"]["process_mb"] - (self.baseline_memory / 1024 / 1024))
        memory_growth_rate_mb_per_hour = (memory_growth_mb / uptime) * 3600 if uptime > 0 else 0

        return {
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "active_sessions": active_sessions,
            "current_snapshot": current_snapshot,
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_mb": round(avg_memory_mb, 2),
                "samples_count": len(self.samples)
            },
            "peaks": {
                "memory_mb": round(max_memory_mb, 2),
                "threads": max_threads,
                "connections": max_connections
            },
            "growth_rates": {
                "memory_mb_per_hour": round(memory_growth_rate_mb_per_hour, 2),
                "memory_total_growth_mb": round(memory_growth_mb, 2)
            },
            "per_session_metrics": {
                "avg_memory_mb": round(avg_memory_mb / active_sessions, 2) if active_sessions > 0 else 0,
                "avg_threads": round(max_threads / active_sessions, 2) if active_sessions > 0 else 0
            },
            "capacity_analysis": self._analyze_capacity(current_snapshot, active_sessions)
        }

    def _analyze_capacity(self, snapshot: Dict, active_sessions: int) -> Dict[str, Any]:
        """Analyze system capacity and bottlenecks"""
        warnings = []
        recommendations = []

        # CPU analysis
        if snapshot["cpu"]["process_percent"] > 80:
            warnings.append("High CPU usage detected")
            recommendations.append("Consider horizontal scaling or optimizing CPU-intensive operations")

        # Memory analysis
        if snapshot["memory"]["system_percent"] > 85:
            warnings.append("High system memory usage")
            recommendations.append("Increase system memory or implement memory limits per session")

        # Connection analysis
        if snapshot["connections"] > 900:
            warnings.append("Approaching connection limit")
            recommendations.append("Implement connection pooling or load balancing")

        # Thread analysis
        if snapshot["threads"] > 500:
            warnings.append("High thread count")
            recommendations.append("Review async implementation and connection handling")

        # Estimate capacity
        if active_sessions > 0:
            memory_per_session = snapshot["memory"]["process_mb"] / active_sessions
            available_memory = snapshot["memory"]["available_mb"]
            estimated_capacity = int(available_memory / memory_per_session) if memory_per_session > 0 else 0
        else:
            estimated_capacity = 0

        return {
            "estimated_max_sessions": estimated_capacity,
            "current_utilization_percent": (active_sessions / estimated_capacity * 100) if estimated_capacity > 0 else 0,
            "warnings": warnings,
            "recommendations": recommendations
        }


def generate_combined_report(cost_analyzer: CostAnalyzer, resource_monitor: ResourceMonitor,
                            sessions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive combined cost and resource report"""
    cost_report = cost_analyzer.generate_cost_report(sessions)
    resource_report = resource_monitor.generate_resource_report(len(sessions))

    # Calculate cost per resource metrics
    cost_per_mb_hour = 0
    cost_per_cpu_hour = 0

    if resource_report["uptime_seconds"] > 0:
        hours = resource_report["uptime_seconds"] / 3600
        if cost_report.get("cost_summary"):
            total_cost = cost_report["cost_summary"]["total_usd"]
            avg_memory = resource_report["averages"]["memory_mb"]
            avg_cpu = resource_report["averages"]["cpu_percent"]

            cost_per_mb_hour = (total_cost / (avg_memory * hours)) if avg_memory > 0 and hours > 0 else 0
            cost_per_cpu_hour = (total_cost / (avg_cpu * hours)) if avg_cpu > 0 and hours > 0 else 0

    return {
        "timestamp": time.time(),
        "report_type": "combined_cost_and_resource_analysis",
        "cost_analysis": cost_report,
        "resource_analysis": resource_report,
        "efficiency_metrics": {
            "cost_per_mb_hour_usd": round(cost_per_mb_hour, 8),
            "cost_per_cpu_percent_hour_usd": round(cost_per_cpu_hour, 6),
            "sessions_per_dollar": round(1 / cost_report["cost_summary"]["avg_per_session_usd"], 2)
            if cost_report.get("cost_summary") and cost_report["cost_summary"]["avg_per_session_usd"] > 0 else 0
        },
        "summary": {
            "total_cost_usd": cost_report.get("cost_summary", {}).get("total_usd", 0),
            "avg_cpu_percent": resource_report["averages"]["cpu_percent"],
            "avg_memory_mb": resource_report["averages"]["memory_mb"],
            "active_sessions": len(sessions),
            "uptime_hours": resource_report["uptime_seconds"] / 3600
        }
    }


def save_report_to_file(report: Dict[str, Any], filename: str = None):
    """Save report to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cost_analysis_report_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    return filename
