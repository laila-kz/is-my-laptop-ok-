"""Collect system performance samples for training the one-class SVM.

Writes a CSV with one row per sample:

    Timestamp,CPU_Usage_Percent,Memory_Usage_Percent,Disk_Usage_Percent

The column order here must stay in sync with kFeatureOrder in
src/train_svm.cpp, otherwise the model will learn on permuted features.

Example:
    python scripts/collect-data.py --duration 7200 --interval 5 \\
        --output data/system_performance_data.csv
"""

import argparse
import csv
import datetime
import pathlib
import sys
import time

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.exit("psutil is required. Install it with: pip install -r requirements.txt")

HEADER = [
    "Timestamp",
    "CPU_Usage_Percent",
    "Memory_Usage_Percent",
    "Disk_Usage_Percent",
]

# '/' resolves to the current drive root on Windows but is ambiguous, so use
# pathlib's anchor to get an explicit path such as 'C:\\'.
DEFAULT_DISK_PATH = str(pathlib.Path(psutil.disk_partitions()[0].mountpoint)
                        if psutil.disk_partitions() else pathlib.Path.home().anchor)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample CPU, memory and disk usage into a CSV for SVM training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=int, default=7200,
                        help="total collection time in seconds")
    parser.add_argument("--interval", type=int, default=5,
                        help="seconds between samples")
    parser.add_argument("--output", default="data/system_performance_data.csv",
                        help="destination CSV file")
    parser.add_argument("--disk-path", default=DEFAULT_DISK_PATH,
                        help="filesystem to measure disk usage on")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.interval < 1:
        sys.exit("--interval must be at least 1 second.")
    if args.duration < args.interval:
        sys.exit("--duration must be >= --interval.")

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_samples = args.duration // args.interval
    print(f"Collecting {total_samples} samples every {args.interval}s "
          f"({args.duration // 60} min) -> {output}")
    print("Press Ctrl+C to stop early; rows collected so far are kept.\n")

    start = time.monotonic()
    collected = 0

    # Prime psutil so the first interval-based reading is meaningful.
    psutil.cpu_percent(interval=0.2)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)

        while collected < total_samples:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory().percent
                disk = psutil.disk_usage(args.disk_path).percent
            except (OSError, psutil.Error) as error:
                print(f"Skipping sample: {error}", file=sys.stderr)
            else:
                writer.writerow([timestamp, cpu, memory, disk])
                handle.flush()
                collected += 1
                if collected % 10 == 0 or collected == total_samples:
                    done = (time.monotonic() - start) / max(args.duration, 1)
                    print(f"  {collected}/{total_samples} samples "
                          f"({done * 100:5.1f}% of the run)")

            # Sleep only the remainder so sampling stays on schedule.
            target = start + (collected + 1) * args.interval
            time.sleep(max(target - time.monotonic(), 0))

    print(f"\nDone. {collected} samples written to {output}")
    print("Next step: train_svm.exe --csv " + str(output).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; partial dataset kept.")
        sys.exit(130)
