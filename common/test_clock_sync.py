import tempfile
import unittest
from unittest.mock import patch
import queue

from common.clock_sync import ClockOffsetEstimator, SAMPLE_INTERVAL_SECONDS


class ClockOffsetEstimatorTests(unittest.TestCase):
    def test_estimate_uses_fifth_percentile_after_minimum_samples(self):
        estimator = ClockOffsetEstimator(["imu"])

        for index in range(9):
            node_time = 1000 + index * 20
            estimator.observe_sample("imu", node_time, node_time / 1000 + 0.5)
        self.assertIsNone(estimator.get_offsets()["imu"])

        for index in range(9, 40):
            node_time = 1000 + index * 20
            offset = 0.1 + index * 0.01
            estimator.observe_sample("imu", node_time, node_time / 1000 + offset)

        expected = sorted(
            [0.5] * 9 + [0.1 + index * 0.01 for index in range(9, 40)]
        )[int((40 - 1) * 0.05)]
        self.assertAlmostEqual(estimator.get_offsets()["imu"], expected)

    def test_freezing_stops_offset_updates_until_reset(self):
        estimator = ClockOffsetEstimator(["imu"])
        for index in range(10):
            node_time = 1000 + index * 20
            estimator.observe_sample("imu", node_time, node_time / 1000 + 0.25)

        frozen_offset = estimator.freeze()["imu"]
        for index in range(10, 20):
            node_time = 1000 + index * 20
            estimator.observe_sample("imu", node_time, node_time / 1000 + 2.0)

        self.assertEqual(estimator.get_offsets()["imu"], frozen_offset)
        estimator.reset_observations()
        self.assertIsNone(estimator.get_offsets()["imu"])

    def test_unwraps_millis_rollover(self):
        estimator = ClockOffsetEstimator(["imu"])
        before_rollover = estimator.unwrap_sample("imu", 0xFFFFFFF0)
        after_rollover = estimator.unwrap_sample("imu", 4)
        self.assertAlmostEqual(after_rollover - before_rollover, 0.020)


class PredictionSyncEngineTests(unittest.TestCase):
    def test_waits_for_every_device_then_freezes_shared_offset_estimates(self):
        from prediction_system.src.system.sync_engine import SyncEngine

        devices = ["imu_1", "imu_2"]
        engine = SyncEngine(queue.Queue(), queue.Queue(), devices)

        for index in range(10):
            node_time = 1000 + index * 50
            node_elapsed = node_time / 1000
            engine._ingest_imu({
                "node_id": devices[0],
                "node_time": node_time,
                "sync_time": 50.0 + node_elapsed + 0.2,
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
            })
        self.assertFalse(engine._lock_clock_offsets())

        for index in range(9):
            node_time = 1000 + index * 50
            node_elapsed = node_time / 1000
            engine._ingest_imu({
                "node_id": devices[1],
                "node_time": node_time,
                "sync_time": 51.0 + node_elapsed + 0.4,
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
            })
        self.assertFalse(engine._lock_clock_offsets())

        node_time = 1450
        node_elapsed = node_time / 1000
        engine._ingest_imu({
            "node_id": devices[1],
            "node_time": node_time,
            "sync_time": 51.0 + node_elapsed + 0.4,
            "ax": 0.0, "ay": 0.0, "az": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0,
        })
        self.assertTrue(engine._lock_clock_offsets())
        self.assertAlmostEqual(engine.clock_offsets[devices[0]], 50.2)
        self.assertAlmostEqual(engine.clock_offsets[devices[1]], 51.4)
        self.assertEqual(engine.frame_interval, SAMPLE_INTERVAL_SECONDS)

        engine._ingest_imu({
            "node_id": devices[0],
            "node_time": 1500,
            "sync_time": 100.0,
            "ax": 0.0, "ay": 0.0, "az": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0,
        })
        self.assertAlmostEqual(engine.clock_offsets[devices[0]], 50.2)


class DatasetRecorderClockTests(unittest.TestCase):
    def test_recorder_uses_shared_offsets_and_maps_samples_to_pc_time(self):
        from data_collection.src.system.dataset_recorder import DatasetRecorder

        with tempfile.TemporaryDirectory() as recordings_dir:
            with patch("data_collection.src.system.dataset_recorder.os.makedirs"):
                recorder = DatasetRecorder(["imu"])
            recorder.save_dir = recordings_dir

            for index in range(10):
                node_time = 1000 + index * 50
                node_elapsed = node_time / 1000
                arrival_time = 500.0 + node_elapsed
                recorder.record_imu_sample({
                    "node_id": "imu",
                    "node_time": node_time,
                    "arrival_sync_time": arrival_time,
                    "sync_time": arrival_time,
                })

            self.assertEqual(recorder.get_clock_offsets()["imu"], 500.0)
            recorder.start_recording("clock_test")
            mapped_time = recorder.record_imu_sample({
                "node_id": "imu",
                "node_time": 1500,
                "arrival_sync_time": 501.5,
                "sync_time": 501.5,
                "ax": 0.0, "ay": 0.0, "az": 0.0,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
            })

            self.assertAlmostEqual(mapped_time, 501.5)
            self.assertAlmostEqual(
                recorder.imu_samples["imu"][0]["sync_time"], 501.5
            )


if __name__ == "__main__":
    unittest.main()
