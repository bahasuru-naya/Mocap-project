from collections import deque


MINIMUM_CLOCK_SAMPLES = 10
CLOCK_OBSERVATION_WINDOW = 500
CLOCK_OFFSET_PERCENTILE = 0.05
SAMPLE_RATE_HZ = 20
SAMPLE_INTERVAL_SECONDS = 1.0 / SAMPLE_RATE_HZ
UINT32_RANGE = 1 << 32
UINT32_HALF_RANGE = 1 << 31


class ClockOffsetEstimator:
    """Estimate per-device PC-monotonic offsets from IMU arrival observations."""

    def __init__(
        self,
        target_devices,
        minimum_samples=MINIMUM_CLOCK_SAMPLES,
        observation_window=CLOCK_OBSERVATION_WINDOW,
        percentile=CLOCK_OFFSET_PERCENTILE,
    ):
        self.target_devices = tuple(target_devices)
        self.minimum_samples = minimum_samples
        self.percentile = percentile
        self.observations = {
            device: deque(maxlen=observation_window)
            for device in self.target_devices
        }
        self._clock_state = {
            device: (None, 0) for device in self.target_devices
        }
        self._frozen_offsets = None

    def observe_sample(self, device, node_time, arrival_time):
        """Unwrap a node timestamp and collect an offset until calibration freezes."""
        node_elapsed = self.unwrap_sample(device, node_time)
        if node_elapsed is not None and self._frozen_offsets is None:
            self.observations[device].append(arrival_time - node_elapsed)
        return node_elapsed

    def unwrap_sample(self, device, node_time):
        if device not in self._clock_state or node_time is None:
            return None

        raw_node_ms = int(node_time) % UINT32_RANGE
        previous_raw_ms, uptime_epoch_ms = self._clock_state[device]
        if previous_raw_ms is not None and raw_node_ms < previous_raw_ms:
            backwards = previous_raw_ms - raw_node_ms
            if backwards > UINT32_HALF_RANGE:
                uptime_epoch_ms += UINT32_RANGE
            elif backwards > 1000:
                uptime_epoch_ms = 0
                self.observations[device].clear()

        self._clock_state[device] = (raw_node_ms, uptime_epoch_ms)
        return (uptime_epoch_ms + raw_node_ms) / 1000.0

    def get_offsets(self):
        if self._frozen_offsets is not None:
            return dict(self._frozen_offsets)

        return {
            device: self._estimate(device)
            for device in self.target_devices
        }

    def freeze(self):
        offsets = self.get_offsets()
        if any(offset is None for offset in offsets.values()):
            raise RuntimeError("Clock offsets are not ready for every device.")
        self._frozen_offsets = offsets
        return dict(offsets)

    def reset_observations(self):
        """Begin a new pre-recording calibration without losing uptime tracking."""
        self._frozen_offsets = None
        for samples in self.observations.values():
            samples.clear()

    def sample_counts(self):
        return {
            device: len(samples)
            for device, samples in self.observations.items()
        }

    def _estimate(self, device):
        samples = self.observations[device]
        if len(samples) < self.minimum_samples:
            return None
        sorted_samples = sorted(samples)
        index = int((len(sorted_samples) - 1) * self.percentile)
        return sorted_samples[index]
