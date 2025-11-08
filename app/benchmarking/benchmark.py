import requests
import time
import argparse
import csv
import json
import os
import statistics
from tqdm import tqdm

def clear_server_history(server_url):
    """Calls the /clear_history endpoint to reset the server's state."""
    clear_url = server_url.replace('/generate_accompaniment', '/clear_history')
    try:
        response = requests.post(clear_url)
        response.raise_for_status()
        print("Successfully cleared server history.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error clearing server history: {e}")
        return False

def inject_music_to_server(server_url: str, injection_file_path: str, injection_length_ticks: int):
    """
    Inject music into the server's inference engine history.
    
    Args:
        server_url: URL of the generate_accompaniment endpoint
        injection_file_path: Path to MIDI file to inject
        injection_length_ticks: Number of ticks to inject from the file
    
    Returns:
        int: Number of ticks actually injected, 0 if failed
    """
    injection_url = server_url.replace('/generate_accompaniment', '/inject_music')
    
    try:
        request_data = {
            "injection_file_path": injection_file_path,
            "injection_length_ticks": injection_length_ticks
        }
        
        print(f"🎵 Injecting music: {injection_file_path} ({injection_length_ticks} ticks)")
        response = requests.post(injection_url, json=request_data)
        response.raise_for_status()
        
        result = response.json()
        if result['success']:
            print(f"✅ Injection successful: {result['melody_notes_injected']} melody notes, {result['accompaniment_notes_injected']} accompaniment notes")
            return result['injection_length_ticks']
        else:
            print(f"❌ Injection failed: {result['message']}")
            return 0
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during injection: {e}")
        return 0

def run_benchmark(server_url: str, num_requests: int, output_file: str, generation_length_frames: int = None, 
                 tempo: float = None, assumed_network_latency_ms: float = None, inference_interval_ticks: int = None,
                 prompt_length_ticks: int = None, injection_file_path: str = None, injection_length_ticks: int = None):
    """
    Runs the benchmark by sending a fixed set of notes to the server multiple times.

    Args:
        server_url (str): The full URL to the /generate_accompaniment endpoint.
        num_requests (int): The number of requests to send.
        output_file (str): The path to save the resulting CSV file.
        generation_length_frames (int): Optional generation length to test specific values.
        tempo (float): BPM for musical timing analysis.
        assumed_network_latency_ms (float): Additional network latency to include in analysis.
        inference_interval_ticks (int): Specific tick interval to analyze.
        prompt_length_ticks (int): Optional context length in ticks for the model.
        injection_file_path (str): Path to MIDI file to inject as prompt.
        injection_length_ticks (int): Number of ticks to inject from the file.
    """
    # A consistent set of notes to send for each request to ensure a fair test.
    # This simulates playing a C major scale.
    melody_notes = [
        {
          "pitch": 64,
          "tick": 0,
          "duration": 4
        },
        {
          "pitch": 62,
          "tick": 4,
          "duration": 4
        },
        {
          "pitch": 60,
          "tick": 8,
          "duration": 4
        },
        {
          "pitch": 62,
          "tick": 12,
          "duration": 4
        }
    ]
    # We will always ask the model to generate notes starting from the next bar.
    base_generation_start_tick = 16

    # Prepare to collect data
    all_results = []
    all_responses = []  # Store complete response data
    
    print(f"Starting benchmark: {num_requests} requests to {server_url}")

    # Reset the server state before we begin.
    if not clear_server_history(server_url):
        print("Halting benchmark due to server connection issue.")
        return
    
    # Inject music if specified
    injection_offset_ticks = 0
    if injection_file_path and injection_length_ticks:
        injection_offset_ticks = inject_music_to_server(server_url, injection_file_path, injection_length_ticks)
        if injection_offset_ticks == 0:
            print("Injection failed. Halting benchmark.")
            return
        print(f"🎯 Injection offset: {injection_offset_ticks} ticks. Adjusting generation_start_tick.")

    for i in tqdm(range(num_requests), desc="Benchmarking", unit="req"):
        
        # Adjust generation start tick based on injection offset
        adjusted_generation_start_tick = base_generation_start_tick + injection_offset_ticks
        
        request_data = {
            "melody_notes": melody_notes,
            "generation_start_tick": adjusted_generation_start_tick,
            "client_request_send_time": time.perf_counter()
        }
        
        # Add generation length if specified
        if generation_length_frames is not None:
            request_data["generation_length_frames"] = generation_length_frames
            
        # Add timing analysis parameters if specified
        if tempo is not None:
            request_data["tempo"] = tempo
        if assumed_network_latency_ms is not None:
            request_data["assumed_network_latency_ms"] = assumed_network_latency_ms
        if inference_interval_ticks is not None:
            request_data["inference_interval_ticks"] = inference_interval_ticks
        if prompt_length_ticks is not None:
            request_data["prompt_length_ticks"] = prompt_length_ticks

        try:
            # --- Send Request and Measure Time ---
            start_time = time.perf_counter()
            response = requests.post(server_url, json=request_data)
            end_time = time.perf_counter()
            
            response.raise_for_status()
            response_json = response.json()

            # --- Calculate Latencies ---
            round_trip_time = end_time - start_time
            
            timings = response_json['timings']
            server_arrival = timings['request_arrival_time']
            server_response = timings['response_output_time']
            server_processing_duration = server_response - server_arrival
            total_network_latency = round_trip_time - server_processing_duration
            
            # Calculate detailed server-side durations
            preprocess_duration = timings['inference_start_time'] - timings['preprocess_start_time']
            inference_duration = timings['inference_end_time'] - timings['inference_start_time']
            postprocess_duration = timings['response_output_time'] - timings['postprocess_start_time']

            result = {
                'request_id': i + 1,
                'num_generated_notes': len(response_json['accompaniment']),
                'generation_start_tick': response_json['generation_start_tick'],
                
                # Client-side measurements
                'round_trip_time': round_trip_time,
                'total_network_latency': total_network_latency,
                
                # Server-side timing (raw timestamps)
                'server_request_arrival_time': timings['request_arrival_time'],
                'server_response_output_time': timings['response_output_time'],
                'server_preprocess_start_time': timings['preprocess_start_time'],
                'server_inference_start_time': timings['inference_start_time'],
                'server_inference_end_time': timings['inference_end_time'],
                'server_postprocess_start_time': timings['postprocess_start_time'],
                
                # Server-side durations (calculated from same clock)
                'server_processing_duration': server_processing_duration,
                'preprocess_duration': preprocess_duration,
                'inference_duration': inference_duration,
                'postprocess_duration': postprocess_duration,
            }
            
            # Store complete response data for JSON output
            response_record = {
                'request_id': i + 1,
                'request_data': request_data,
                'response_data': response_json,
                'client_timing': {
                    'request_start_time': start_time,
                    'request_end_time': end_time,
                    'round_trip_time': round_trip_time,
                    'total_network_latency': total_network_latency,
                    'server_processing_duration': server_processing_duration,
                    'preprocess_duration': preprocess_duration,
                    'inference_duration': inference_duration,
                    'postprocess_duration': postprocess_duration,
                }
            }
            
            all_results.append(result)
            all_responses.append(response_record)
            
            # Brief pause to avoid overwhelming the server
            time.sleep(0.1)

        except requests.exceptions.HTTPError as http_err:
            print(f"  Request {i+1} failed: {http_err}")
            try:
                # FastAPI provides detailed validation errors in the JSON response
                print(f"  Server validation error: {http_err.response.json()}")
            except ValueError:
                # If the response isn't JSON, print the raw text
                print(f"  Server response: {http_err.response.text}")
            continue
        except requests.exceptions.RequestException as req_err:
            print(f"  Request {i+1} failed: {req_err}")
            continue
    
    # --- Save Results ---
    if not all_results:
        print("No successful requests. No data to save.")
        return

    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save CSV with timing summary
    with open(output_file, 'w', newline='') as f:
        header = all_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)

    # Save JSON with complete response data
    json_output_file = output_file.replace('.csv', '.json')
    benchmark_data = {
        'metadata': {
            'server_url': server_url,
            'num_requests': num_requests,
            'total_successful_requests': len(all_results),
            'melody_notes_sent': melody_notes,
            'generation_start_tick': adjusted_generation_start_tick,
            'generation_length_frames': generation_length_frames,
            'tempo': tempo,
            'assumed_network_latency_ms': assumed_network_latency_ms,
            'inference_interval_ticks': inference_interval_ticks,
            'benchmark_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'responses': all_responses
    }
    
    with open(json_output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nBenchmark finished.")
    print(f"Timing summary saved to: {output_file}")
    print(f"Complete response data saved to: {json_output_file}")
    
    # --- Print Summary Statistics ---
    round_trips = [r['round_trip_time'] for r in all_results]
    server_processing = [r['server_processing_duration'] for r in all_results]
    inference_times = [r['inference_duration'] for r in all_results]
    preprocess_times = [r['preprocess_duration'] for r in all_results]
    postprocess_times = [r['postprocess_duration'] for r in all_results]
    network_latencies = [r['total_network_latency'] for r in all_results]
    num_notes = [r['num_generated_notes'] for r in all_results]

    print("\n--- Summary Statistics ---")
    print(f"Successful Requests: {len(all_results)}/{num_requests}")
    print(f"Generated Notes per Request: {statistics.mean(num_notes):.1f} (std: {statistics.stdev(num_notes):.1f})")
    print()
    print("Timing Breakdown:")
    print(f"  Round Trip Time:       {statistics.mean(round_trips)*1000:.1f}ms (std: {statistics.stdev(round_trips)*1000:.1f}ms)")
    print(f"  Server Processing:     {statistics.mean(server_processing)*1000:.1f}ms (std: {statistics.stdev(server_processing)*1000:.1f}ms)")
    print(f"    - Preprocessing:     {statistics.mean(preprocess_times)*1000:.1f}ms (std: {statistics.stdev(preprocess_times)*1000:.1f}ms)")
    print(f"    - Inference:         {statistics.mean(inference_times)*1000:.1f}ms (std: {statistics.stdev(inference_times)*1000:.1f}ms)")
    print(f"    - Postprocessing:    {statistics.mean(postprocess_times)*1000:.1f}ms (std: {statistics.stdev(postprocess_times)*1000:.1f}ms)")
    print(f"  Network Latency:       {statistics.mean(network_latencies)*1000:.1f}ms (std: {statistics.stdev(network_latencies)*1000:.1f}ms)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the StreamMUSE server.")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8000/generate_accompaniment",
        help="The URL of the server's /generate_accompaniment endpoint."
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=100,
        help="The number of requests to send for the benchmark."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output CSV file (e.g., 'results/my_test.csv')."
    )
    parser.add_argument(
        "--generation_length_frames",
        type=int,
        default=None,
        help="Number of frames to generate per request (overrides server default)."
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=None,
        help="BPM for musical timing analysis."
    )
    parser.add_argument(
        "--assumed_network_latency_ms",
        type=float,
        default=None,
        help="Additional network latency to include in analysis (ms)."
    )
    parser.add_argument(
        "--inference_interval_ticks",
        type=int,
        default=None,
        help="Specific tick interval to analyze."
    )
    parser.add_argument(
        "--prompt_length_ticks",
        type=int,
        default=None,
        help="Context length in ticks for the model (overrides server default)."
    )
    parser.add_argument(
        "--injection_file_path",
        type=str,
        default=None,
        help="Path to MIDI file to inject as prompt before benchmarking."
    )
    parser.add_argument(
        "--injection_length_ticks",
        type=int,
        default=None,
        help="Number of ticks to inject from the file."
    )
    args = parser.parse_args()

    run_benchmark(args.server_url, args.num_requests, args.output_file, args.generation_length_frames,
                  args.tempo, args.assumed_network_latency_ms, args.inference_interval_ticks, args.prompt_length_ticks,
                  args.injection_file_path, args.injection_length_ticks) 