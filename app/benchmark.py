import requests
import time
import argparse
import csv
import json
import os
import statistics

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

def run_benchmark(server_url: str, num_requests: int, output_file: str):
    """
    Runs the benchmark by sending a fixed set of notes to the server multiple times.

    Args:
        server_url (str): The full URL to the /generate_accompaniment endpoint.
        num_requests (int): The number of requests to send.
        output_file (str): The path to save the resulting CSV file.
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
    generation_start_tick = 16

    # Prepare to collect data
    all_results = []
    all_responses = []  # Store complete response data
    
    print(f"Starting benchmark: {num_requests} requests to {server_url}")

    # Reset the server state before we begin.
    if not clear_server_history(server_url):
        print("Halting benchmark due to server connection issue.")
        return

    for i in range(num_requests):
        print(f"Sending request {i+1}/{num_requests}...")
        
        request_data = {
            "melody_notes": melody_notes,
            "generation_start_tick": generation_start_tick,
            "client_request_send_time": time.perf_counter()
        }

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
            'generation_start_tick': generation_start_tick,
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
    args = parser.parse_args()

    run_benchmark(args.server_url, args.num_requests, args.output_file) 