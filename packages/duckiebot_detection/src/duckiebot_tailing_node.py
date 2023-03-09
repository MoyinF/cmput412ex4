#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped
from duckietown_msgs.srv import ChangePattern
from dt_apriltags import Detector, Detection
import yaml

ROAD_MASK = [(20, 60, 0), (50, 255, 255)] # for yellow mask
DEBUG = False
ENGLISH = False

# using code from Justin's lane follower: works but sometimes goes straight off the mat...
# problem: cropping sometimes removes the yellow contours, also I think the offset is too high

class DuckiebotTailingNode(DTROS):

    def __init__(self, node_name):
        super(DuckiebotTailingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Publishers
        self.pub_mask = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed", CompressedImage, queue_size=1)
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_img_bool = True

        # Subscribers
        self.sub_camera = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",CompressedImage,self.img_callback,queue_size=1,buff_size="20MB")
        self.sub_distance = rospy.Subscriber(f'/{self.veh}/duckiebot_distance_node/distance', Float32, self.dist_callback)
        self.sub_detection = rospy.Subscriber(f'/{self.veh}/duckiebot_detection_node/detection', BoolStamped, self.detection_callback)

        # image processing tools
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        self.detection = False
        self.distance = 0
        self.intersection_detected = False

        self.loginfo("Initialized")

        # find the calibration parameters
        camera_intrinsic_dict =  self.readYamlFile(f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')

        self.K = np.array(camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.D = np.array(camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
        self.h = camera_intrinsic_dict["image_height"]
        self.w = camera_intrinsic_dict["image_width"]

        f_x = camera_intrinsic_dict['camera_matrix']['data'][0]
        f_y = camera_intrinsic_dict['camera_matrix']['data'][4]
        c_x = camera_intrinsic_dict['camera_matrix']['data'][2]
        c_y = camera_intrinsic_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]

        # initialize apriltag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
        self.intersections = {
            133: 'RED', # T intersection
            153: 'RED', # T intersection
            62: 'RED', # T intersection
            58: 'RED', # T intersection
        }
        self.led_colors = {
            0: 'WHITE',
            1: 'RED'
        }

        # apriltag detection filters
        self.decision_threshold = 10
        self.z_threshold = 0.17

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -170
        else:
            self.offset = 170
        self.varying_velocity = 0.25
        self.velocity = 0.25
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.025
        self.D = -0.0025
        self.last_error = 0
        self.last_time = rospy.get_time()

        self.forward_P = 0.005
        self.forward_D = -0.001
        self.target = 0.025
        self.forward_prop = 0
        self.last_fwd_err = 0
        self.last_distance = 0

        # Service proxies
        # rospy.wait_for_service(f'/{self.veh}/led_emitter_node/set_pattern')
        # self.led_service = rospy.ServiceProxy(f'/{self.veh}/led_emitter_node/set_pattern', ChangePattern)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def run(self):
        if self.detection:
            # P Term
            self.forward_prop = self.distance - self.target
            P = -self.forward_prop * self.forward_P

            # D Term
            d_error = (self.forward_prop - self.last_fwd_err) / (rospy.get_time() - self.last_time)
            self.last_fwd_err = self.forward_prop
            self.last_time = rospy.get_time()
            D = d_error * self.forward_D

            if self.distance <= self.last_distance:
                self.twist.v = 0
                self.twist.omega = 0
            elif self.distance > self.last_distance:
                # if the duckiebot is lagging behind the leader, speed up
                if self.distance > self.target:
                    # self.varying_velocity = self.varying_velocity + 0.01
                    self.twist.v = self.varying_velocity
                    self.twist.omega = 0
            self.last_distance = self.distance
            self.vel_pub.publish(self.twist)

            '''
            elif self.forward_prop <= 0:
                self.varying_velocity = self.varying_velocity - 0.01
                self.twist.v = self.varying_velocity
                self.twist.omega = 0
            elif self.forward_prop > 0:
                self.twist.v = self.velocity
                self.twist.omega = 0
            '''

            ################## deal with turns later
            # self.twist.omega = P + D

        else:
            self.drive()

    def dist_callback(self, msg):
        self.distance = msg.data

    def detection_callback(self, msg):
        self.detection = msg.data

    def img_callback(self, msg):
        img = self.jpeg.decode(msg.data)
        self.intersection_detected = self.detect_intersection(msg)
        crop = img[:, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        # debugging
        if self.pub_img_bool:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)
        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

    def drive(self):
        if self.intersection_detected:
            self.intersection_sequence()
        else:
            if self.proportional is None:
                self.twist.omega = 0
            else:
                # P Term
                P = -self.proportional * self.P

                # D Term
                d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
                self.last_error = self.proportional
                self.last_time = rospy.get_time()
                D = d_error * self.D

                self.twist.v = self.velocity
                self.twist.omega = P + D
                if DEBUG:
                    self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

            self.vel_pub.publish(self.twist)

    def intersection_sequence(self):
        # for now
        rospy.loginfo("detected intersection")
        # self.changeLED(1)
        # rospy.sleep(5)
        # self.changeLED(0)
        self.drive()

    def detect_intersection(self, img_msg):
        cv_image = None
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except CvBridgeError as e:
            self.log(e)
            return []

        # undistort the image
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.w, self.h), 0, (self.w, self.h))
        image_np = cv2.undistort(cv_image, self.K, self.D, None, newcameramtx)

        # convert the image to black and white
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # detect tags present in image
        tags = self.at_detector.detect(image_gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065)

        closest_tag_z = 1000
        closest = None

        for tag in tags:
            # ignore distant tags and tags with bad decoding
            z = tag.pose_t[2][0]
            if tag.decision_margin < self.decision_threshold or z > self.z_threshold:
                continue

            # update the closest-detected tag if needed
            if z < closest_tag_z:
                closest_tag_z = z
                closest = tag

        if closest:
            if closest.tag_id in self.intersections:
                return True
        return False

    def changeLED(self, tag_id):
        color = self.led_colors[tag_id]
        try:
            self.led_service(String(color))
        except Exception as e:
            rospy.loginfo("Failed to publish LEDs: " + str(e))

    def detect_intersection_with_reds(self, img):
        img = img[300:-1, :, :]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([160, 50, 50])
        upper_range = np.array([180, 255,255])
        red_lower_range = np.array([0, 95, 153])
        red_upper_range = np.array([7, 244,255])
        lower_mask = cv2.inRange(hsv, lower_range, upper_range)
        upper_mask = cv2.inRange(hsv, red_lower_range, red_upper_range)
        full_mask = lower_mask + upper_mask
        edged = cv2.Canny(full_mask, 30, 200)
        contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        return max_idx != -1

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)

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


if __name__ == "__main__":
    node = DuckiebotTailingNode("duckiebot_tailing_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()
