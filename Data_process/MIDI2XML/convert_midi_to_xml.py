#!/usr/bin/env python3
"""
MIDI to MusicXML converter using MuseScore command line tool.
Uses multiprocessing to accelerate conversion.
"""

import os
import subprocess
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_single_midi(midi_path: str, output_dir: str, timeout: int = 120) -> tuple[str, bool, str]:
    """
    Convert a single MIDI file to MusicXML using MuseScore.

    Args:
        midi_path: Path to the input MIDI file
        output_dir: Directory to save the output MusicXML file
        timeout: Timeout in seconds for the conversion process

    Returns:
        Tuple of (midi_path, success, error_message)
    """
    try:
        midi_file = Path(midi_path)
        output_file = Path(output_dir) / f"{midi_file.stem}.musicxml"

        # Skip if output already exists
        if output_file.exists():
            return (midi_path, True, "Already exists")

        # Run MuseScore in offscreen mode
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'

        cmd = [
            'mscore',
            '-o', str(output_file),
            str(midi_path)
        ]

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Check if conversion was successful
        if output_file.exists() and output_file.stat().st_size > 0:
            return (midi_path, True, "")
        else:
            error_msg = result.stderr if result.stderr else "Output file not created or empty"
            return (midi_path, False, error_msg)

    except subprocess.TimeoutExpired:
        print("timeout!")
        return (midi_path, False, f"Timeout after {timeout}s")
    except Exception as e:
        return (midi_path, False, str(e))


def convert_midi_to_xml(
    input_dir: str,
    output_dir: str,
    num_workers: int = None,
    timeout: int = 120
):
    """
    Convert all MIDI files in a directory to MusicXML using multiprocessing.

    Args:
        input_dir: Directory containing MIDI files
        output_dir: Directory to save MusicXML files
        num_workers: Number of worker processes (default: CPU count)
        timeout: Timeout per file in seconds
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all MIDI files
    midi_files = list(input_path.glob("*.mid")) + list(input_path.glob("*.midi"))
    total_files = len(midi_files)

    if total_files == 0:
        logger.warning(f"No MIDI files found in {input_dir}")
        return

    logger.info(f"Found {total_files} MIDI files to convert")

    # Set number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()

    logger.info(f"Using {num_workers} worker processes")

    # Create partial function with fixed arguments
    convert_func = partial(
        convert_single_midi,
        output_dir=str(output_path),
        timeout=timeout
    )

    # Process files with multiprocessing
    success_count = 0
    fail_count = 0
    skip_count = 0
    failed_files = []

    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(convert_func, [str(f) for f in midi_files]),
            total=total_files,
            desc="Converting MIDI to MusicXML"
        ))

    # Collect statistics
    for midi_path, success, error_msg in results:
        if success:
            if error_msg == "Already exists":
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            failed_files.append((midi_path, error_msg))

    # Log summary
    logger.info(f"\nConversion complete:")
    logger.info(f"  - Successful: {success_count}")
    logger.info(f"  - Skipped (already exists): {skip_count}")
    logger.info(f"  - Failed: {fail_count}")

    # Save failed files to log
    if failed_files:
        fail_log_path = output_path / "conversion_failures.log"
        with open(fail_log_path, 'w') as f:
            for midi_path, error_msg in failed_files:
                f.write(f"{midi_path}\t{error_msg}\n")
        logger.info(f"Failed files logged to: {fail_log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MIDI files to MusicXML using MuseScore"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default="/DATA6_6T/cby/musicxml/lmd_muti_filtered",
        help="Directory containing MIDI files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/DATA6_6T/cby/musicxml/lmd_muti_xml",
        help="Directory to save MusicXML files"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=60,
        help="Timeout per file in seconds (default: 60)"
    )

    args = parser.parse_args()

    convert_midi_to_xml(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.workers,
        timeout=args.timeout
    )


if __name__ == "__main__":
    main()
