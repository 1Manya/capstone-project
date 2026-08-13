#!/usr/bin/env python3
"""
nuscenes_publisher.py
======================
Plays back ONE nuScenes Mini scene as live ROS 2 topics:
    /sensors/camera   (sensor_msgs/Image)      - CAM_FRONT
    /sensors/lidar    (sensor_msgs/PointCloud2) - LIDAR_TOP
    /sensors/radar    (sensor_msgs/PointCloud2) - RADAR_FRONT
    /sensors/imu      (sensor_msgs/Imu)         - derived from ego_pose (placeholder,
                                                   see note in the safety blueprint's
                                                   decisions log re: CAN bus expansion)

Usage:
    python3 nuscenes_publisher.py --dataroot ~/data/nuscenes --scene-idx 0 --rate 2.0
"""

import argparse
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, Imu
from std_msgs.msg import Header

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud


def make_image_msg(img_bgr: np.ndarray, frame_id: str, stamp) -> Image:
    """Build a sensor_msgs/Image manually - NOT using cv_bridge, to avoid a
    known compatibility issue between pip's opencv-python and the system
    cv_bridge's internal type table (KeyError on cvtype_to_name lookups).
    This is functionally identical to what cv_bridge would produce for a
    standard bgr8 OpenCV image."""
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height, msg.width, channels = img_bgr.shape
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = channels * img_bgr.shape[1]
    msg.data = np.ascontiguousarray(img_bgr).tobytes()
    return msg


def make_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """points_xyz: (N, 3) float32 array -> PointCloud2 message."""
    header = Header(frame_id=frame_id, stamp=stamp)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    data = points_xyz.astype(np.float32).tobytes()
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = points_xyz.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 12  # 3 * float32
    msg.row_step = msg.point_step * points_xyz.shape[0]
    msg.is_dense = True
    msg.data = data
    return msg


class NuScenesPublisher(Node):
    def __init__(self, dataroot: str, scene_idx: int, rate_hz: float):
        super().__init__("nuscenes_publisher")

        self.cam_pub = self.create_publisher(Image, "/sensors/camera", 10)
        self.lidar_pub = self.create_publisher(PointCloud2, "/sensors/lidar", 10)
        self.radar_pub = self.create_publisher(PointCloud2, "/sensors/radar", 10)
        self.imu_pub = self.create_publisher(Imu, "/sensors/imu", 10)

        self.get_logger().info(f"Loading nuScenes Mini from {dataroot} ...")
        self.nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

        scene = self.nusc.scene[scene_idx]
        self.get_logger().info(f"Playing scene: {scene['name']} - {scene['description']}")
        self.sample_token = scene["first_sample_token"]

        self.prev_translation = None
        self.prev_time = None

        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self.publish_next_sample)

    def publish_next_sample(self):
        if self.sample_token == "":
            self.get_logger().info("Scene finished. Stopping.")
            self.timer.cancel()
            return

        sample = self.nusc.get("sample", self.sample_token)
        stamp = self.get_clock().now().to_msg()

        # ---- Camera (CAM_FRONT) ----
        cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cam_path = self.nusc.get_sample_data_path(cam_data["token"])
        img = cv2.imread(cam_path)
        if img is not None:
            img_msg = make_image_msg(img, "camera", stamp)
            self.cam_pub.publish(img_msg)

        # ---- LiDAR (LIDAR_TOP) ----
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_path = self.nusc.get_sample_data_path(lidar_data["token"])
        pc = LidarPointCloud.from_file(lidar_path)
        xyz = pc.points[:3, :].T  # (N, 3)
        self.lidar_pub.publish(make_pointcloud2(xyz, "lidar", stamp))

        # ---- Radar (RADAR_FRONT) ----
        if "RADAR_FRONT" in sample["data"]:
            radar_data = self.nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
            radar_path = self.nusc.get_sample_data_path(radar_data["token"])
            rpc = RadarPointCloud.from_file(radar_path)
            rxyz = rpc.points[:3, :].T
            self.radar_pub.publish(make_pointcloud2(rxyz, "radar", stamp))

        # ---- IMU (placeholder, derived from ego_pose translation delta) ----
        # NOTE: this is a simplification, NOT real IMU data. nuScenes Mini's
        # standard download doesn't include raw IMU - that lives in the
        # separate "CAN bus expansion" download. Documented as a known
        # limitation in the safety blueprint's decisions log.
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        translation = np.array(ego_pose["translation"])
        cur_time = ego_pose["timestamp"] / 1e6  # microseconds -> seconds

        imu_msg = Imu()
        imu_msg.header = Header(frame_id="imu", stamp=stamp)
        if self.prev_translation is not None and self.prev_time is not None:
            dt = max(cur_time - self.prev_time, 1e-3)
            velocity = (translation - self.prev_translation) / dt
            imu_msg.linear_acceleration.x = float(velocity[0])
            imu_msg.linear_acceleration.y = float(velocity[1])
            imu_msg.linear_acceleration.z = float(velocity[2])
        self.imu_pub.publish(imu_msg)
        self.prev_translation = translation
        self.prev_time = cur_time

        self.get_logger().info(f"Published sample {sample['token'][:8]}...")
        self.sample_token = sample["next"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/manya_singh/data/nuscenes")
    ap.add_argument("--scene-idx", type=int, default=0)
    ap.add_argument("--rate", type=float, default=2.0, help="Hz - nuScenes keyframes are ~2Hz natively")
    args = ap.parse_args()

    rclpy.init()
    node = NuScenesPublisher(args.dataroot, args.scene_idx, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
