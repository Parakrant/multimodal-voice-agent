"""
Performance Testing and Benchmarking Script

Tests the multi-modal pipeline under various concurrency levels
and generates performance reports.
"""

import asyncio
import json
import time
import statistics
from typing import List, Dict, Any
import websockets
from datetime import datetime
import argparse


class PerformanceBenchmark:
    """Benchmark the multi-modal voice agent pipeline"""

    def __init__(self, ws_url: str = "ws://localhost:8000/ws/multimodal"):
        self.ws_url = ws_url
        self.results: List[Dict[str, Any]] = []

    async def single_text_turn(self, test_text: str) -> Dict[str, Any]:
        """Execute a single text turn and measure performance"""
        start_time = time.time()
        metrics = {
            "start_time": start_time,
            "frames_received": [],
            "errors": [],
            "latencies": {}
        }

        try:
            async with websockets.connect(self.ws_url) as ws:
                # Receive init
                init_msg = await ws.recv()
                init_data = json.loads(init_msg)
                metrics["session_id"] = init_data.get("session_id")
                metrics["frames_received"].append(("init", time.time() - start_time))

                # Send user text
                send_time = time.time()
                await ws.send(json.dumps({
                    "type": "user_text",
                    "text": test_text
                }))

                # Collect all response frames
                ack_received = False
                llm_complete = False
                tts_complete = False
                viz_complete = False
                audio_received = False

                while not all([llm_complete, tts_complete, viz_complete]):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)

                        if isinstance(msg, bytes):
                            # Audio data
                            audio_received = True
                            metrics["audio_size_bytes"] = len(msg)
                            metrics["latencies"]["audio_received"] = time.time() - send_time
                            continue

                        data = json.loads(msg)
                        frame_type = data.get("type")
                        frame_time = time.time() - send_time

                        metrics["frames_received"].append((frame_type, frame_time))

                        if frame_type == "ack":
                            ack_received = True
                            metrics["latencies"]["ack"] = frame_time

                        elif frame_type == "llm_start":
                            metrics["latencies"]["llm_start"] = frame_time

                        elif frame_type == "llm_complete":
                            llm_complete = True
                            metrics["latencies"]["llm_complete"] = frame_time
                            metrics["assistant_text"] = data.get("text", "")

                        elif frame_type == "tool_call_start":
                            metrics["latencies"]["tool_call_start"] = frame_time
                            metrics["tool_calls"] = data.get("tool_calls", [])

                        elif frame_type == "tool_call_complete":
                            metrics["latencies"]["tool_call_complete"] = frame_time

                        elif frame_type == "viz_complete":
                            viz_complete = True
                            metrics["latencies"]["viz_complete"] = frame_time

                        elif frame_type == "tts_start":
                            metrics["latencies"]["tts_start"] = frame_time

                        elif frame_type == "tts_complete":
                            tts_complete = True
                            metrics["latencies"]["tts_complete"] = frame_time

                        elif frame_type == "visualization_data":
                            metrics["latencies"]["visualization_data"] = frame_time

                        elif frame_type == "cost_update":
                            metrics["cost"] = data.get("cost_breakdown", {})

                        elif frame_type == "metrics_update":
                            metrics["server_latency"] = data.get("latency")

                        elif frame_type == "error":
                            metrics["errors"].append(data.get("error"))

                    except asyncio.TimeoutError:
                        metrics["errors"].append("Timeout waiting for response")
                        break

                metrics["total_latency"] = time.time() - start_time
                metrics["success"] = len(metrics["errors"]) == 0

        except Exception as e:
            metrics["errors"].append(str(e))
            metrics["success"] = False

        return metrics

    async def concurrent_load_test(self, num_concurrent: int, test_text: str) -> Dict[str, Any]:
        """Run concurrent connections test"""
        print(f"\n{'='*60}")
        print(f"Running load test with {num_concurrent} concurrent connections")
        print(f"{'='*60}")

        start_time = time.time()

        tasks = [
            self.single_text_turn(test_text)
            for _ in range(num_concurrent)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_duration = end_time - start_time

        # Process results
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and not r.get("success"))]

        latencies = [r["total_latency"] for r in successful if "total_latency" in r]
        llm_latencies = [r["latencies"]["llm_complete"] for r in successful if "latencies" in r and "llm_complete" in r["latencies"]]
        tts_latencies = [r["latencies"]["tts_complete"] for r in successful if "latencies" in r and "tts_complete" in r["latencies"]]

        report = {
            "test_config": {
                "num_concurrent": num_concurrent,
                "test_text": test_text,
                "test_duration_sec": total_duration
            },
            "results": {
                "total_requests": num_concurrent,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate_percent": (len(successful) / num_concurrent) * 100
            },
            "latency_stats": self._calculate_stats(latencies, "total"),
            "llm_latency_stats": self._calculate_stats(llm_latencies, "llm"),
            "tts_latency_stats": self._calculate_stats(tts_latencies, "tts"),
            "throughput": {
                "requests_per_second": num_concurrent / total_duration if total_duration > 0 else 0,
                "avg_concurrent_processing_time": statistics.mean(latencies) if latencies else 0
            }
        }

        # Cost analysis
        costs = [r.get("cost", {}).get("total", 0) for r in successful if "cost" in r]
        if costs:
            report["cost_analysis"] = {
                "total_cost_usd": sum(costs),
                "avg_cost_per_request_usd": statistics.mean(costs),
                "cost_per_second_usd": sum(costs) / total_duration if total_duration > 0 else 0
            }

        return report

    def _calculate_stats(self, values: List[float], label: str) -> Dict[str, float]:
        """Calculate statistical metrics"""
        if not values:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "stddev": 0,
                "p95": 0,
                "p99": 0
            }

        sorted_values = sorted(values)
        return {
            "count": len(values),
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "stddev": round(statistics.stdev(values), 3) if len(values) > 1 else 0,
            "p95": round(sorted_values[int(len(sorted_values) * 0.95)], 3),
            "p99": round(sorted_values[int(len(sorted_values) * 0.99)], 3)
        }

    async def run_benchmark_suite(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print(f"\n{'#'*60}")
        print("MULTI-MODAL VOICE AGENT PIPELINE BENCHMARK")
        print(f"{'#'*60}")
        print(f"Start Time: {datetime.now()}")
        print(f"WebSocket URL: {self.ws_url}")

        all_results = []

        for test_case in test_cases:
            num_concurrent = test_case.get("concurrent", 1)
            test_text = test_case.get("text", "Hello, this is a test message.")

            result = await self.concurrent_load_test(num_concurrent, test_text)
            all_results.append(result)

            # Print summary
            print(f"\nConcurrency: {num_concurrent}")
            print(f"  Success Rate: {result['results']['success_rate_percent']:.1f}%")
            print(f"  Avg Latency: {result['latency_stats']['mean']:.3f}s")
            print(f"  P95 Latency: {result['latency_stats']['p95']:.3f}s")
            print(f"  P99 Latency: {result['latency_stats']['p99']:.3f}s")
            print(f"  Throughput: {result['throughput']['requests_per_second']:.2f} req/s")

            # Wait between tests
            await asyncio.sleep(2)

        # Generate final report
        final_report = {
            "timestamp": time.time(),
            "benchmark_suite": "multi_modal_voice_agent",
            "test_results": all_results,
            "summary": self._generate_summary(all_results)
        }

        return final_report

    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary of all benchmark results"""
        total_requests = sum(r["results"]["total_requests"] for r in results)
        total_successful = sum(r["results"]["successful"] for r in results)
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0

        all_latencies = []
        for r in results:
            if r["latency_stats"]["count"] > 0:
                # Approximate latencies from mean and count
                all_latencies.extend([r["latency_stats"]["mean"]] * r["latency_stats"]["count"])

        return {
            "total_requests": total_requests,
            "total_successful": total_successful,
            "overall_success_rate_percent": round(overall_success_rate, 2),
            "overall_avg_latency_sec": round(statistics.mean(all_latencies), 3) if all_latencies else 0,
            "max_concurrency_tested": max(r["test_config"]["num_concurrent"] for r in results),
            "recommendations": self._generate_recommendations(results)
        }

    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Check success rates
        for result in results:
            if result["results"]["success_rate_percent"] < 95:
                concurrency = result["test_config"]["num_concurrent"]
                recommendations.append(
                    f"Success rate dropped to {result['results']['success_rate_percent']:.1f}% "
                    f"at {concurrency} concurrent connections. Consider horizontal scaling."
                )

        # Check latencies
        for result in results:
            if result["latency_stats"]["p95"] > 5.0:
                recommendations.append(
                    f"P95 latency exceeded 5 seconds at {result['test_config']['num_concurrent']} "
                    "concurrent connections. Investigate bottlenecks in STT/LLM/TTS pipeline."
                )

        # Check throughput
        max_throughput = max(r["throughput"]["requests_per_second"] for r in results)
        if max_throughput < 10:
            recommendations.append(
                f"Maximum throughput is {max_throughput:.2f} req/s. "
                "Consider implementing connection pooling and optimizing async operations."
            )

        if not recommendations:
            recommendations.append(
                "System performance is excellent across all concurrency levels tested."
            )

        return recommendations


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Multi-Modal Voice Agent Pipeline")
    parser.add_argument("--url", default="ws://localhost:8000/ws/multimodal",
                       help="WebSocket URL")
    parser.add_argument("--max-concurrency", type=int, default=20,
                       help="Maximum concurrency to test")
    parser.add_argument("--output", default="benchmark_report.json",
                       help="Output file for results")

    args = parser.parse_args()

    # Define test cases (conservative for ElevenLabs rate limits)
    test_cases = [
        {"concurrent": 1, "text": "Hello, tell me about the weather."},
        {"concurrent": 2, "text": "Can you create a chart showing sales data?"},
        {"concurrent": 3, "text": "Analyze the sentiment of this message: I love this product!"},
        {"concurrent": 5, "text": "Get me real-time stock market data."},
        {"concurrent": min(args.max_concurrency, 10), "text": "What are the current usage statistics?"}
    ]

    benchmark = PerformanceBenchmark(ws_url=args.url)

    try:
        report = await benchmark.run_benchmark_suite(test_cases)

        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Benchmark Complete!")
        print(f"{'='*60}")
        print(f"\nSUMMARY:")
        print(f"  Total Requests: {report['summary']['total_requests']}")
        print(f"  Overall Success Rate: {report['summary']['overall_success_rate_percent']}%")
        print(f"  Overall Avg Latency: {report['summary']['overall_avg_latency_sec']}s")
        print(f"  Max Concurrency Tested: {report['summary']['max_concurrency_tested']}")
        print(f"\nRECOMMENDATIONS:")
        for rec in report['summary']['recommendations']:
            print(f"  - {rec}")
        print(f"\nFull report saved to: {args.output}")

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
