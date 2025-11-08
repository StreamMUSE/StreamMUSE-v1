#!/usr/bin/env python3
"""
Helper script for manual generation length studies.

This script helps coordinate manual testing of different generation lengths
by providing easy commands and tracking which tests have been completed.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import time

class ManualGenerationLengthStudy:
    """
    Helps coordinate manual testing of different generation lengths.
    """
    
    def __init__(self, output_dir: str = "results/manual_gen_length_study"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.output_dir / "study_status.json"
        self.load_status()
    
    def load_status(self):
        """Load study status from file."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {
                'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'completed_lengths': [],
                'failed_lengths': [],
                'total_requests': 0
            }
    
    def save_status(self):
        """Save study status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def show_status(self):
        """Display current study status."""
        print("📊 Manual Generation Length Study Status")
        print("=" * 50)
        print(f"Study Directory: {self.output_dir.absolute()}")
        print(f"Created: {self.status['created']}")
        print(f"Total Requests Completed: {self.status['total_requests']}")
        
        if self.status['completed_lengths']:
            print(f"✅ Completed Generation Lengths: {sorted(self.status['completed_lengths'])}")
        
        if self.status['failed_lengths']:
            print(f"❌ Failed Generation Lengths: {sorted(self.status['failed_lengths'])}")
        
        print("\n" + "=" * 50)
    
    def get_server_command(self, generation_length: int, port: int = 8000) -> str:
        """Get the server command for a specific generation length."""
        return (f"GENERATION_LENGTH_FRAMES={generation_length} "
                f"uvicorn app.server:app --host 0.0.0.0 --port {port}")
    
    def get_benchmark_command(self, generation_length: int, num_requests: int = 50, 
                            server_url: str = "http://localhost:8000/generate_accompaniment") -> str:
        """Get the benchmark command for a specific generation length."""
        output_file = self.output_dir / f"gen_length_{generation_length}.csv"
        return (f"python app/benchmark.py --server_url {server_url} "
                f"--num_requests {num_requests} --output_file {output_file}")
    
    def run_single_test(self, generation_length: int, num_requests: int = 50,
                       server_url: str = "http://localhost:8000/generate_accompaniment") -> bool:
        """Run a single benchmark test for the specified generation length."""
        
        if generation_length in self.status['completed_lengths']:
            print(f"⚠️  Generation length {generation_length} already completed")
            return True
        
        print(f"\n🧪 Testing Generation Length: {generation_length} frames")
        print("=" * 50)
        
        # Check if server is running
        if not self._test_server_connection(server_url):
            print(f"❌ Cannot connect to server at {server_url}")
            print(f"Please start server with: {self.get_server_command(generation_length)}")
            return False
        
        # Run benchmark
        output_file = self.output_dir / f"gen_length_{generation_length}.csv"
        
        benchmark_cmd = [
            sys.executable, "app/benchmark.py",
            "--server_url", server_url,
            "--num_requests", str(num_requests),
            "--output_file", str(output_file)
        ]
        
        print(f"Running: {' '.join(benchmark_cmd)}")
        
        try:
            result = subprocess.run(benchmark_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ Benchmark failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                self.status['failed_lengths'].append(generation_length)
                self.save_status()
                return False
            
            # Verify output file
            if not output_file.exists():
                print(f"❌ Output file not created: {output_file}")
                self.status['failed_lengths'].append(generation_length)
                self.save_status()
                return False
            
            print(f"✅ Benchmark completed for generation length {generation_length}")
            self.status['completed_lengths'].append(generation_length)
            self.status['total_requests'] += num_requests
            self.save_status()
            return True
            
        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timed out")
            self.status['failed_lengths'].append(generation_length)
            self.save_status()
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            self.status['failed_lengths'].append(generation_length)
            self.save_status()
            return False
    
    def _test_server_connection(self, server_url: str) -> bool:
        """Test if server is responding."""
        import requests
        try:
            clear_url = server_url.replace('/generate_accompaniment', '/clear_history')
            response = requests.post(clear_url, timeout=5)
            return response.status_code in [200, 503]
        except:
            return False
    
    def analyze_results(self):
        """Run analysis on completed results."""
        if not self.status['completed_lengths']:
            print("❌ No completed tests to analyze")
            return False
        
        print(f"\n📊 Analyzing results from {len(self.status['completed_lengths'])} tests...")
        
        analysis_cmd = [
            sys.executable, "app/analyze_generation_length_results.py",
            str(self.output_dir),
            "--output_dir", str(self.output_dir / "analysis")
        ]
        
        try:
            result = subprocess.run(analysis_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Analysis failed:")
                print(f"STDERR: {result.stderr}")
                return False
            
            print("✅ Analysis completed!")
            print(result.stdout)
            return True
            
        except Exception as e:
            print(f"❌ Error running analysis: {e}")
            return False
    
    def generate_instructions(self, generation_lengths: list, num_requests: int = 50):
        """Generate step-by-step instructions for manual testing."""
        
        instructions_file = self.output_dir / "testing_instructions.md"
        
        with open(instructions_file, 'w') as f:
            f.write("# Manual Generation Length Testing Instructions\n\n")
            f.write(f"**Study Directory:** `{self.output_dir.absolute()}`\n\n")
            f.write(f"**Generation Lengths to Test:** {generation_lengths}\n")
            f.write(f"**Requests per Test:** {num_requests}\n\n")
            
            f.write("## Step-by-Step Process\n\n")
            
            for i, gen_length in enumerate(generation_lengths, 1):
                f.write(f"### Step {i}: Test Generation Length {gen_length}\n\n")
                f.write("1. **Stop the current server** (Ctrl+C)\n\n")
                f.write("2. **Start server with new generation length:**\n")
                f.write(f"   ```bash\n")
                f.write(f"   {self.get_server_command(gen_length)}\n")
                f.write(f"   ```\n\n")
                f.write("3. **Wait for server to load** (\"Inference engine loaded successfully\")\n\n")
                f.write("4. **Run benchmark:**\n")
                f.write(f"   ```bash\n")
                f.write(f"   {self.get_benchmark_command(gen_length, num_requests)}\n")
                f.write(f"   ```\n\n")
                f.write("5. **Verify completion** (should see timing statistics)\n\n")
                
                if i < len(generation_lengths):
                    f.write("6. **Continue to next generation length**\n\n")
                else:
                    f.write("6. **All tests complete!**\n\n")
            
            f.write("## Analysis\n\n")
            f.write("After completing all tests, run:\n")
            f.write("```bash\n")
            f.write(f"python app/analyze_generation_length_results.py {self.output_dir}\n")
            f.write("```\n\n")
            
            f.write("## Alternative: Use Helper Script\n\n")
            f.write("You can also run individual tests using:\n")
            f.write("```bash\n")
            f.write("python app/manual_generation_length_study.py test <generation_length>\n")
            f.write("```\n\n")
            f.write("Or check status with:\n")
            f.write("```bash\n")
            f.write("python app/manual_generation_length_study.py status\n")
            f.write("```\n")
        
        print(f"📝 Instructions saved to: {instructions_file}")
        return instructions_file


def main():
    parser = argparse.ArgumentParser(
        description="Manual generation length study helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status                    Show current study status
  test <gen_length>         Run test for specific generation length
  instructions <lengths>    Generate testing instructions
  analyze                   Analyze completed results

Examples:
  %(prog)s status
  %(prog)s test 20
  %(prog)s instructions 5,10,15,20,25,30
  %(prog)s analyze
        """
    )
    
    parser.add_argument("command", choices=['status', 'test', 'instructions', 'analyze'],
                       help="Command to execute")
    parser.add_argument("argument", nargs='?',
                       help="Argument for command (generation length or lengths list)")
    parser.add_argument("--output_dir", default="results/manual_gen_length_study",
                       help="Output directory for study")
    parser.add_argument("--num_requests", type=int, default=50,
                       help="Number of requests per test")
    parser.add_argument("--server_url", default="http://localhost:8000/generate_accompaniment",
                       help="Server URL")
    
    args = parser.parse_args()
    
    study = ManualGenerationLengthStudy(args.output_dir)
    
    if args.command == 'status':
        study.show_status()
        
    elif args.command == 'test':
        if not args.argument:
            print("❌ Please specify generation length to test")
            return 1
        
        try:
            gen_length = int(args.argument)
        except ValueError:
            print("❌ Generation length must be an integer")
            return 1
        
        study.show_status()
        success = study.run_single_test(gen_length, args.num_requests, args.server_url)
        study.show_status()
        
        return 0 if success else 1
        
    elif args.command == 'instructions':
        if not args.argument:
            print("❌ Please specify generation lengths (e.g., '5,10,15,20')")
            return 1
        
        try:
            generation_lengths = [int(x.strip()) for x in args.argument.split(',')]
        except ValueError:
            print("❌ Invalid generation lengths format")
            return 1
        
        study.generate_instructions(generation_lengths, args.num_requests)
        
    elif args.command == 'analyze':
        success = study.analyze_results()
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())