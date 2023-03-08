#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
from dt_class_utils import DTReminder
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String, Header, Float32, ColorRGBA
from duckietown_msgs.msg import LEDPattern, WheelsCmdStamped
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern
import time
from cv_bridge import CvBridge, CvBridgeError
import yaml

class LaneFollowingNode(DTROS):
    def __init__(self):
        super(LaneFollowingNode, self).__init__(node_name="lane_following_node", node_type=NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")

        # Publisher for wheel velocities
        self.pub_wheel_commands = rospy.Publisher(f'/{self.veh_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.image_pub = rospy.Publisher(f'/{self.veh_name}/lane_following_node/lane_detection_image/image/compressed', CompressedImage, queue_size=16)

        # Subscriber to the camera image
        self.sub = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.callback)
        self.image_data = None
        self.bridge = CvBridge()

        # read camera intrinsics
        camera_intrinsic_dict =  self.readYamlFile(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')
        self.K = np.array(camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.D = np.array(camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
        self.h = camera_intrinsic_dict["image_height"]
        self.w = camera_intrinsic_dict["image_width"]

        # self.screen_w = self.w * 2
        # self.screen_h = int(self.h * 2 * (3/4))
        self.screen_w = 640 * 2
        self.screen_h = int(480 * 2 * (3/4))
        self.center = (int(self.screen_w/2), int(self.screen_h))
        self.white_point = self.center
        self.yellow_point = self.center
        self.red_point = self.center
        self.yellow_contours = None
        self.white_contours = None
        self.red_contours = None
        self.any_red = False
        self.yellow_dist = 0
        self.white_dist = 0
        self.red_dist = 0
        self.red_margin = 10

        self.vel = 0.3
        self.p_gain = 0.005
        self.d_gain = 0.001
        self.margin = 10
        self.dt = 1/5
        self.error = 0
        self.last_error = 0
        self.firstRun = True

    def run(self):
        frequency = 5 # 5Hz
        self.dt = 1 / frequency
        rate = rospy.Rate(frequency)
        while not rospy.is_shutdown():
            if self.image_data is not None:
                # undist_image = self.undistort_image(self.image_data)

                # decided to use distorted_image
                cv_image = self.bridge.compressed_imgmsg_to_cv2(self.image_data)
                yellow_line = self.yellow_mask(cv_image)
                white_line = self.white_mask(cv_image)

                red_lower_range = np.array([160, 50, 50])
                red_upper_range = np.array([180, 255,255])
                intersection = False
                red_image = cv_image[500:, :]
                self.any_red = False
                red_line = self.red_mask(cv_image)
                if self.any_red:
                    if abs(self.red_dist) < self.red_margin:
                        rospy.loginfo("detected red")
                        intersection = True

                img_cp = cv_image.copy()
                cv2.drawContours(img_cp, self.white_contours, -1, (0, 255, 0), 3)
                cv2.drawContours(img_cp, self.yellow_contours, -1, (0, 255, 0), 3)
                # make new CompressedImage to publish
                augmented_image_msg = CompressedImage()
                augmented_image_msg.header.stamp = rospy.Time.now()
                augmented_image_msg.format = "jpeg"
                augmented_image_msg.data = np.array(cv2.imencode('.jpg', img_cp)[1]).tostring()

                # Publish new image
                self.image_pub.publish(augmented_image_msg)

                self.move(self.vel, self.vel)
                if intersection:
                    # keep moving forward for 0.5 meters
                    self.forward(0.5)
                else:
                    self.error = self.get_error(abs(self.yellow_dist), abs(self.white_dist))
                    P = abs(self.error) * (self.p_gain)                                     # proportional term
                    if self.firstRun:
                        self.last_error = self.error
                        self.firstRun = False
                    valueRateOfChange = (self.error - self.last_error) / self.dt
                    self.last_error = self.error
                    D = self.d_gain * -valueRateOfChange                                    # derivative term
                    vel_diff = P + D

                    if self.error < -self.margin:
                        # error is yellow - white, so negative error means too much space on white side
                        # for right lane following, this means the right wheel is moving too fast
                        # so we need to speed up the left wheel
                        self.move(self.vel + (vel_diff/2), self.vel - (vel_diff/2))
                    elif self.error > self.margin:
                        self.move(self.vel - (vel_diff/2), self.vel + (vel_diff/2))
                    else:
                        self.move(self.vel, self.vel)

    def callback(self, data):
        self.image_data = data

    def get_error(self, y_dist, w_dist):
        # error is yellow - white
        return y_dist - w_dist

    def move(self, vel_left, vel_right):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.vel_left = vel_left
        msg.vel_right = vel_right
        self.pub_wheel_commands.publish(msg)

    def forward(self, dist):
        # if starting movement at intersection, give initial speed
        # if self.firstRun:
            # self.move(self.vel, self.vel)
        # don't change velocities, just keep moving at current speed
        sec = 3
        rospy.loginfo(sec)
        time.sleep(sec)

    def yellow_mask(self, img):
        # returns mask: a binary image where the white pixels are where yellow occurs in the image
        # hsv color ranges obtained using color threshold program
        # need to readjust values using a clearer picture from the duckiebot camera

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([16, 0, 156])
        upper_range = np.array([40,255,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # Finding Contours
        # with canny edge detection
        edged = cv2.Canny(mask, 30, 200)
        # Use a copy of the image e.g. edged.copy() since findContours alters the image
        self.yellow_contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # remember dist is negative
        dist = -640
        for cnt in self.yellow_contours:
            d = cv2.pointPolygonTest(cnt,self.center, True)
            if d > dist:
                dist = d
        self.yellow_point = (int(self.center[0] + dist), int(self.center[1]))
        self.yellow_dist = dist

        return mask

    def white_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([48, 0, 144])
        upper_range = np.array([142, 81,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # Finding Contours
        # With canny edge detection:
        edged = cv2.Canny(mask, 30, 200)
        # Use a copy of the image e.g. edged.copy() since findContours alters the image
        self.white_contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        dist = -640
        for cnt in self.white_contours:
            d = cv2.pointPolygonTest(cnt,self.center, True)
            if d > dist:
                dist = d
        self.white_point = (int(self.center[0] - dist), int(self.center[1]))
        self.white_dist = dist

        return mask

    def red_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([160, 106, 106])
        upper_range = np.array([180, 255,255])
        red_lower_range = np.array([0, 95, 153])
        red_upper_range = np.array([7, 244,255])

        lower_mask = cv2.inRange(hsv, lower_range, upper_range)
        upper_mask = cv2.inRange(hsv, red_lower_range, red_upper_range)
        full_mask = lower_mask + upper_mask


        # Finding Contours
        # With canny edge detection:
        edged = cv2.Canny(full_mask, 30, 200)
        # Use a copy of the image e.g. edged.copy() since findContours alters the image
        self.red_contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        dist = -640
        for cnt in self.red_contours:
            d = cv2.pointPolygonTest(cnt,self.center, True)
            if d > dist:
                dist = d
        self.red_point = (int(self.center[0]), int(self.center[1] + dist))
        self.red_dist = dist

        if len(self.red_contours) > 0:
            self.any_red = True
        return full_mask

    def undistort_image(self, image_data):
        try:
            # Convert the image to cv2 type
            cv_image = self.bridge.compressed_imgmsg_to_cv2(image_data)

            # undistort the image
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.w, self.h), 0, (self.w, self.h))
            cv_image = cv2.undistort(cv_image, self.K, self.D, None, newcameramtx)

            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def readYamlFile(self,fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def stop(self):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.vel_left = 0.0
        msg.vel_right = 0.0
        self.pub_wheel_commands.publish(msg)

    def on_shutdown(self):
        self.stop()
        rospy.loginfo(f"[{self.node_name}] Shutting down.")

if __name__ == "__main__":
    # create the node
    node = LaneFollowingNode()
    # run node
    node.run()
    rospy.on_shutdown(node.on_shutdown)
    # keep spinning
    rospy.spin()
