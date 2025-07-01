import requests
import time
import argparse
import csv
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
    
    print(f"Starting benchmark: {num_requests} requests to {server_url}")

    # Reset the server state before we begin.
    if not clear_server_history(server_url):
        print("Halting benchmark due to server connection issue.")
        return

    for i in range(num_requests):
        print(f"Sending request {i+1}/{num_requests}...")
        
        request_data = {
            "melody_notes": melody_notes,
            "generation_start_tick": generation_start_tick
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

            all_results.append({
                'round_trip_time': round_trip_time,
                'server_processing_duration': server_processing_duration,
                'total_network_latency': total_network_latency
            })
            
            # Brief pause to avoid overwhelming the server
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"  Request {i+1} failed: {e}")
            continue
    
    # --- Save Results to CSV ---
    if not all_results:
        print("No successful requests. No data to save.")
        return

    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        header = all_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nBenchmark finished. Results saved to {output_file}")
    
    # --- Print Summary Statistics ---
    round_trips = [r['round_trip_time'] for r in all_results]
    server_processing = [r['server_processing_duration'] for r in all_results]

    print("\n--- Summary ---")
    print(f"Successful Requests: {len(all_results)}/{num_requests}")
    print(f"Avg. Round Trip Time:      {statistics.mean(round_trips):.4f}s (std: {statistics.stdev(round_trips):.4f}s)")
    print(f"Avg. Server Processing Time: {statistics.mean(server_processing):.4f}s (std: {statistics.stdev(server_processing):.4f}s)")

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
        default=20,
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