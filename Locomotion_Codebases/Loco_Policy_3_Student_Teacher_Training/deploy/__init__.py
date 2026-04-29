"""Real hardware deployment layer for Boston Dynamics Spot.

spot_sdk_wrapper: Policy I/O <-> Spot SDK bridge (20 Hz control loop)
safety_layer: Hardware safety watchdogs (torque, orientation, velocity, timeout)
height_scan_builder: Depth camera/LiDAR -> 187-dim elevation grid
export_onnx: PyTorch -> ONNX conversion with verification
"""
