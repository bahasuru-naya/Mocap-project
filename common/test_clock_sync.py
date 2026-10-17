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
    def setUp(self):
        from prediction_system.src.system.sync_engine import SyncEngine

        self.engine = SyncEngine(queue.Queue(), queue.Queue(), ["imu"])
        self.engine.clock_offsets["imu"] = 0.0
        self.engine.clock_sync_locked = True

    @staticmethod
    def _sample(node_t, value=1.0):
        return {
            "node_t": node_t,
            "ax": value, "ay": value, "az": value,
            "gx": value, "gy": value, "gz": value,
        }

    def test_uses_50ms_hold_and_interpolation_tolerances(self):
        self.assertEqual(self.engine.max_hold, 0.05)
        self.assertEqual(self.engine.max_interpolation_gap, 0.05)

    def test_interpolates_across_50ms_gap_boundary(self):
        self.engine.imu_buffers["imu"].extend([
            self._sample(1.0, 1.0),
            self._sample(1.05, 2.0),
        ])

        sample = self.engine._interpolate_sample("imu", 1.025)

        self.assertIsNotNone(sample)
        self.assertAlmostEqual(sample["ax"], 1.5)

    def test_rejects_interpolation_across_gap_over_50ms(self):
        self.engine.imu_buffers["imu"].extend([
            self._sample(1.0, 1.0),
            self._sample(1.051, 2.0),
        ])

        self.assertIsNone(self.engine._interpolate_sample("imu", 1.025))

    def test_holds_sample_only_within_50ms_of_target(self):
        self.engine.imu_buffers["imu"].append(self._sample(1.0))
        self.engine.last_arrival["imu"] = 1.0
        self.engine.dead_after = 0.0
        self.engine.max_wait = 0.0

        frame_at_limit = self.engine._build_frame(1.05, 2.0)
        frame_over_limit = self.engine._build_frame(1.051, 2.0)

        self.assertIsNotNone(frame_at_limit["imu"]["imu"])
        self.assertIsNone(frame_over_limit["imu"]["imu"])

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


class DatasetRecorderSyncToleranceTests(unittest.TestCase):
    def setUp(self):
        from data_collection.src.system.dataset_recorder import DatasetRecorder

        with patch("data_collection.src.system.dataset_recorder.os.makedirs"):
            self.recorder = DatasetRecorder(["imu"])

    def test_uses_50ms_sync_tolerance(self):
        self.assertEqual(self.recorder.max_sync_gap, 0.05)

    def test_rejects_camera_pose_more_than_50ms_from_timeline(self):
        pose = [[1.0, 2.0, 3.0]]
        samples = [{"sync_time": 1.051, "pose": pose}]

        self.assertIsNone(self.recorder._nearest_pose(samples, [1.051], 1.0))

    def test_accepts_camera_pose_at_50ms_boundary(self):
        pose = [[1.0, 2.0, 3.0]]
        samples = [{"sync_time": 1.05, "pose": pose}]

        self.assertEqual(self.recorder._nearest_pose(samples, [1.05], 1.0), pose)

    def test_interpolates_imu_at_50ms_gap_boundary(self):
        samples = [
            {
                "sync_time": 1.0,
                "ax": 1.0, "ay": 1.0, "az": 1.0,
                "gx": 1.0, "gy": 1.0, "gz": 1.0,
            },
            {
                "sync_time": 1.05,
                "ax": 2.0, "ay": 2.0, "az": 2.0,
                "gx": 2.0, "gy": 2.0, "gz": 2.0,
            },
        ]

        values = self.recorder._interpolate_imu(
            [sample["sync_time"] for sample in samples], samples, 1.025
        )

        for value in values:
            self.assertAlmostEqual(value, 1.5)

    def test_does_not_interpolate_imu_across_gap_over_50ms(self):
        samples = [
            {
                "sync_time": 1.0,
                "ax": 1.0, "ay": 1.0, "az": 1.0,
                "gx": 1.0, "gy": 1.0, "gz": 1.0,
            },
            {
                "sync_time": 1.06,
                "ax": 2.0, "ay": 2.0, "az": 2.0,
                "gx": 2.0, "gy": 2.0, "gz": 2.0,
            },
        ]

        values = self.recorder._interpolate_imu(
            [sample["sync_time"] for sample in samples], samples, 1.03
        )

        self.assertEqual(values, [1.0] * 6)


if __name__ == "__main__":
    unittest.main()
