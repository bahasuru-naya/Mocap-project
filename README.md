# Mocap-project

Human motion capture using five BLE IMUs and a camera. Data collection writes a
uniform 20 Hz CSV containing synchronized IMU measurements and 3-D pose
landmarks. Collection aligns its recorded streams after capture; live
prediction uses the same pre-stream clock-offset estimate and 20 Hz timeline.
The prediction system can run live from the IMUs or run the trained
ensemble offline over a saved collection CSV.

- [Data collection setup, recording, and CSV format](./data_collection/README_MOCAP.md)
- [Prediction system setup and offline dataset inference](./prediction_system/README.md)
