#!/usr/bin/env python
# Quick recorder for CARLA 0.9.13. Creates a .log recording.
# Run from PythonAPI/examples:
#   python ./quick_record.py --duration 30 --file test1.log

import argparse
import glob
import os
import sys
import time


def _ensure_carla_import():
    try:
        import carla  # noqa: F401
        return
    except Exception:
        pass

    egg_glob = os.path.join('..', 'carla', 'dist',
                             f"carla-*{sys.version_info.major}.{sys.version_info.minor}-"
                             f"{'win-amd64' if os.name == 'nt' else 'linux-x86_64'}.egg")
    matches = glob.glob(egg_glob)
    if matches:
        sys.path.append(matches[0])
        try:
            import carla  # noqa: F401
            return
        except Exception:
            pass

    raise RuntimeError(
        f"CARLA Python API not found for Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Use Python 3.7 (x64) for CARLA 0.9.13 and run from PythonAPI/examples."
    )


def main():
    _ensure_carla_import()
    import carla

    ap = argparse.ArgumentParser(description='Record a CARLA session to a .log file')
    ap.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    ap.add_argument('--port', default=2000, type=int, help='CARLA server port')
    ap.add_argument('--duration', '-d', default=30, type=int, help='Seconds to record')
    ap.add_argument('--file', '-f', default='test1.log', help='Output filename')
    ap.add_argument('--sensors', action='store_true', help='Include sensors in recording')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    print(f'Recording to {args.file} for {args.duration}s (sensors={args.sensors})...')
    client.start_recorder(args.file, args.sensors)
    try:
        time.sleep(args.duration)
    finally:
        client.stop_recorder()
        print('Recording stopped.')


if __name__ == '__main__':
    main()
