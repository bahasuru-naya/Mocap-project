"""Measure BLE IMU throughput and packet loss.

Flash IMU_BLE_Performance_Test.ino, install ``data_collection/requirements.txt``,
then run, for example:

    python ble_performance_test.py --rate 20 --duration 120
    python ble_performance_test.py --rate 50 --duration 120

Each run writes per-sample CSV data and timestamped JSONL events to ``logs/``.
The ESP32 also prints five-second sensor-read and BLE-send counters to Serial.
"""

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from bleak import BleakClient, BleakScanner


TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
CSV_FIELDS = [
    "sequence",
    "device_millis",
    "host_time_utc",
    "host_epoch",
    "interarrival_ms",
    "missing_before",
    "record_bytes",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class PerformanceLogger:
    def __init__(self, device_name, target_rate, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"{device_name}_{stamp}"
        self.csv_path = output_dir / f"{stem}.csv"
        self.events_path = output_dir / f"{stem}_events.jsonl"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.csv_writer.writeheader()
        self.events_file = self.events_path.open("w", encoding="utf-8")

        self.device_name = device_name
        self.target_rate = target_rate
        self.started_at = time.monotonic()
        self.connected_at = None
        self.connected_seconds = 0.0
        self.connect_count = 0
        self.received_records = 0
        self.received_notifications = 0
        self.notification_bytes = 0
        self.missing_sequences = 0
        self.duplicate_sequences = 0
        self.out_of_order_sequences = 0
        self.malformed_records = 0
        self.oversized_frames = 0
        self.incomplete_frames = 0
        self.first_sample_at = None
        self.last_sample_at = None
        self.last_sequence = None
        self.last_sample_monotonic = None
        self.intervals = deque(maxlen=10000)
        self.interval_sum = 0.0
        self.interval_count = 0
        self.window_started_at = time.monotonic()
        self.window_records = 0
        self.window_bytes = 0

    def event(self, event, **details):
        item = {"time_utc": utc_now(), "event": event, **details}
        self.events_file.write(json.dumps(item, separators=(",", ":")) + "\n")
        self.events_file.flush()
        print(f"[{item['time_utc']}] {event}: {details}")

    def on_connect(self, rssi=None, mtu=None):
        self.connect_count += 1
        self.connected_at = time.monotonic()
        self.last_sequence = None
        self.event("connected", connection_number=self.connect_count, rssi=rssi, mtu=mtu)

    def on_disconnect(self, reason=None):
        now = time.monotonic()
        if self.connected_at is not None:
            self.connected_seconds += now - self.connected_at
            self.connected_at = None
        self.last_sequence = None
        self.event("disconnected", reason=reason)

    def notification(self, notification_bytes):
        self.received_notifications += 1
        self.notification_bytes += len(notification_bytes)
        self.window_bytes += len(notification_bytes)

    def malformed(self, reason, raw_line):
        self.malformed_records += 1
        self.event("malformed_record", reason=reason, raw=raw_line.decode("utf-8", "replace"))

    def sample(self, line):
        try:
            values = line.decode("ascii").strip().split(",")
            if len(values) != 8:
                raise ValueError(f"expected 8 fields, received {len(values)}")
            sequence = int(values[0])
            device_millis = int(values[1])
            axes = [float(value) for value in values[2:]]
            if not 0 <= sequence <= 0xFFFFFFFF or not 0 <= device_millis <= 0xFFFFFFFF:
                raise ValueError("sequence and device time must be unsigned 32-bit integers")
            if not all(math.isfinite(value) for value in axes):
                raise ValueError("sensor value is not finite")
        except (UnicodeDecodeError, ValueError) as error:
            self.malformed(str(error), line)
            return

        now = time.monotonic()
        host_epoch = time.time()
        interarrival_ms = None
        missing_before = 0
        if self.last_sample_monotonic is not None:
            interval = now - self.last_sample_monotonic
            interarrival_ms = interval * 1000.0
            self.intervals.append(interarrival_ms)
            self.interval_sum += interval
            self.interval_count += 1

        if self.last_sequence is not None:
            difference = (sequence - self.last_sequence) & 0xFFFFFFFF
            if difference == 0:
                self.duplicate_sequences += 1
            elif difference < 0x80000000:
                missing_before = difference - 1
                self.missing_sequences += missing_before
            else:
                self.out_of_order_sequences += 1

        if self.first_sample_at is None:
            self.first_sample_at = now
        self.last_sample_at = now
        self.last_sample_monotonic = now
        self.last_sequence = sequence
        self.received_records += 1
        self.window_records += 1
        self.csv_writer.writerow(
            {
                "sequence": sequence,
                "device_millis": device_millis,
                "host_time_utc": utc_now(),
                "host_epoch": f"{host_epoch:.6f}",
                "interarrival_ms": "" if interarrival_ms is None else f"{interarrival_ms:.3f}",
                "missing_before": missing_before,
                "record_bytes": len(line) + 1,
                "ax": axes[0],
                "ay": axes[1],
                "az": axes[2],
                "gx": axes[3],
                "gy": axes[4],
                "gz": axes[5],
            }
        )
        if self.received_records % 100 == 0:
            self.csv_file.flush()

    def report_interval(self):
        now = time.monotonic()
        elapsed = now - self.window_started_at
        rate = self.window_records / elapsed if elapsed else 0.0
        byte_rate = self.window_bytes / elapsed if elapsed else 0.0
        intervals = list(self.intervals)
        p95 = statistics.quantiles(intervals, n=20)[18] if len(intervals) >= 20 else None
        self.event(
            "interval_stats",
            seconds=round(elapsed, 3),
            records=self.window_records,
            records_per_second=round(rate, 3),
            notification_bytes=self.window_bytes,
            bytes_per_second=round(byte_rate, 1),
            notifications_total=self.received_notifications,
            missing_sequence_total=self.missing_sequences,
            malformed_total=self.malformed_records,
            interarrival_mean_ms=(
                round(statistics.fmean(intervals), 3) if intervals else None
            ),
            interarrival_p95_ms=round(p95, 3) if p95 is not None else None,
            interarrival_max_ms=max(intervals) if intervals else None,
        )
        self.window_started_at = now
        self.window_records = 0
        self.window_bytes = 0

    def summary(self):
        now = time.monotonic()
        connected_seconds = self.connected_seconds
        if self.connected_at is not None:
            connected_seconds += now - self.connected_at
        observed_seconds = (
            self.last_sample_at - self.first_sample_at
            if self.first_sample_at is not None and self.last_sample_at is not None
            else 0.0
        )
        observed_rate = (
            (self.received_records - 1) / observed_seconds
            if observed_seconds > 0 and self.received_records > 1
            else 0.0
        )
        expected_during_connection = round(self.target_rate * connected_seconds)
        estimated_missing = max(0, expected_during_connection - self.received_records)
        interval_values = list(self.intervals)
        summary = {
            "device": self.device_name,
            "target_rate_hz": self.target_rate,
            "run_seconds": round(now - self.started_at, 3),
            "connected_seconds": round(connected_seconds, 3),
            "connections": self.connect_count,
            "received_records": self.received_records,
            "observed_rate_hz_between_first_last_samples": round(observed_rate, 3),
            "expected_records_estimate": expected_during_connection,
            "estimated_missing_from_target_rate": estimated_missing,
            "estimated_loss_percent": (
                round(estimated_missing / expected_during_connection * 100, 3)
                if expected_during_connection
                else None
            ),
            "missing_sequence_numbers": self.missing_sequences,
            "duplicate_sequences": self.duplicate_sequences,
            "out_of_order_sequences": self.out_of_order_sequences,
            "malformed_records": self.malformed_records,
            "oversized_frames": self.oversized_frames,
            "incomplete_frames": self.incomplete_frames,
            "notifications_received": self.received_notifications,
            "notification_bytes": self.notification_bytes,
            "notification_bytes_per_connected_second": (
                round(self.notification_bytes / connected_seconds, 1)
                if connected_seconds
                else 0.0
            ),
            "mean_interarrival_ms": (
                round(statistics.fmean(interval_values), 3) if interval_values else None
            ),
            "max_interarrival_ms": max(interval_values) if interval_values else None,
            "csv_log": str(self.csv_path),
            "events_log": str(self.events_path),
        }
        self.event("final_summary", **summary)
        return summary

    def close(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.events_file.close()


async def run_test_session(args, logger):
    stop_at = time.monotonic() + args.duration if args.duration else None
    connected_once = False

    while stop_at is None or time.monotonic() < stop_at:
        device = None
        client = None
        buffer = bytearray()
        try:
            scan_started = time.monotonic()
            logger.event("scanning_started", device=args.device)
            device = await BleakScanner.find_device_by_filter(
                lambda candidate, advertisement: (
                    candidate.name == args.device
                    or advertisement.local_name == args.device
                ),
                timeout=args.scan_timeout,
            )
            scan_seconds = time.monotonic() - scan_started
            if device is None:
                logger.event("device_not_found", scan_seconds=round(scan_seconds, 3))
                await asyncio.sleep(args.reconnect_delay)
                continue

            logger.event(
                "device_found",
                scan_seconds=round(scan_seconds, 3),
                rssi=getattr(device, "rssi", None),
                address=getattr(device, "address", None),
            )
            connect_started = time.monotonic()
            client = BleakClient(device, timeout=args.connect_timeout)
            await client.connect()
            logger.on_connect(
                rssi=getattr(device, "rssi", None),
                mtu=getattr(client, "mtu_size", None),
            )
            await client.start_notify(
                TX_UUID,
                make_notification_handler(logger, buffer),
            )
            connected_once = True
            logger.event(
                "notify_subscribed",
                connect_seconds=round(time.monotonic() - connect_started, 3),
                mtu=getattr(client, "mtu_size", None),
            )
            await client.write_gatt_char(RX_UUID, f"RATE,{args.rate}".encode("ascii"), response=False)
            logger.event("rate_command_sent", target_rate_hz=args.rate)

            next_report = time.monotonic() + args.report_interval
            while client.is_connected and (stop_at is None or time.monotonic() < stop_at):
                await asyncio.sleep(0.1)
                if time.monotonic() >= next_report:
                    logger.report_interval()
                    next_report = time.monotonic() + args.report_interval
            logger.on_disconnect("BLE link closed")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.event("connection_error", error=repr(error))
            if client is not None and logger.connected_at is not None:
                logger.on_disconnect(repr(error))
        finally:
            if buffer:
                logger.incomplete_frames += 1
                logger.event("incomplete_frame_on_disconnect", buffered_bytes=len(buffer))
            if client is not None and client.is_connected:
                try:
                    await client.disconnect()
                except Exception as error:
                    logger.event("disconnect_error", error=repr(error))

        if stop_at is None or time.monotonic() < stop_at:
            await asyncio.sleep(args.reconnect_delay)

    if not connected_once:
        logger.event("warning", message="No successful BLE connection was made")
    summary = logger.summary()
    return summary


async def run_test(args):
    logger = PerformanceLogger(args.device, args.rate, args.output_dir)
    try:
        return await run_test_session(args, logger)
    except asyncio.CancelledError:
        if logger.connected_at is not None:
            logger.on_disconnect("test interrupted")
        logger.summary()
        raise
    finally:
        logger.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure ESP32 BMI160 BLE throughput, packet loss, and timing."
    )
    parser.add_argument("--device", default="ESP32_IMU_TEST", help="BLE advertised name")
    parser.add_argument("--rate", type=int, default=20, help="requested sensor rate in Hz (1-200)")
    parser.add_argument("--duration", type=float, default=60, help="test duration in seconds; 0 runs until Ctrl+C")
    parser.add_argument("--report-interval", type=float, default=5, help="periodic report interval in seconds")
    parser.add_argument("--scan-timeout", type=float, default=8, help="BLE scan timeout in seconds")
    parser.add_argument("--connect-timeout", type=float, default=15, help="BLE connection timeout in seconds")
    parser.add_argument("--reconnect-delay", type=float, default=2, help="delay before retrying a connection")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="directory for CSV sample log and JSONL event log",
    )
    return parser


async def async_main(args):
    try:
        return await run_test(args)
    except KeyboardInterrupt:
        return None


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.rate <= 200:
        parser.error("--rate must be between 1 and 200 Hz")
    if args.duration < 0 or args.report_interval <= 0:
        parser.error("--duration must be non-negative and --report-interval must be positive")
    try:
        summary = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        summary = None
    if summary:
        print("\nFinal performance summary:")
        print(json.dumps(summary, indent=2))


def make_notification_handler(logger, buffer):
    def handler(_sender, data):
        logger.notification(data)
        buffer.extend(data)
        while b"\n" in buffer:
            line, _, remainder = buffer.partition(b"\n")
            buffer[:] = remainder
            if line:
                logger.sample(line)
        if len(buffer) > 512:
            logger.oversized_frames += 1
            logger.event("oversized_frame", buffered_bytes=len(buffer))
            buffer.clear()

    return handler


if __name__ == "__main__":
    main()
