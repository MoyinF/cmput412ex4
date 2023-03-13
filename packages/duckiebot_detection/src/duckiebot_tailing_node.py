#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped, VehicleCorners
from duckietown_msgs.srv import ChangePattern
from dt_apriltags import Detector, Detection
import yaml

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]  # for yellow mask
DEBUG = False
ENGLISH = False


class DuckiebotTailingNode(DTROS):

    def __init__(self, node_name):
        super(DuckiebotTailingNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Publishers
        self.pub_mask = rospy.Publisher(
            "/" + self.veh + "/output/image/mask/compressed", CompressedImage, queue_size=1)
        self.vel_pub = rospy.Publisher(
            "/" + self.veh + "/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_img_bool = True

        # Subscribers
        self.sub_camera = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                           CompressedImage, self.img_callback, queue_size=1, buff_size="20MB")
        self.sub_distance = rospy.Subscriber(
            f'/{self.veh}/duckiebot_distance_node/distance', Float32, self.dist_callback)
        self.sub_detection = rospy.Subscriber(
            f'/{self.veh}/duckiebot_detection_node/detection', BoolStamped, self.detection_callback)
        self.sub_centers = rospy.Subscriber(f'/{self.veh}/duckiebot_detection_node/centers', VehicleCorners, self.centers_callback, queue_size=1)

        # image processing tools
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        # info from subscribers
        self.detection = False
        self.intersection_detected = False
        self.centers = None

        # find the calibration parameters
        # for detecting apriltags
        camera_intrinsic_dict = self.readYamlFile(
            f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')

        self.K = np.array(
            camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(
            camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.D = np.array(
            camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(
            camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
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
            133: 'INTER',  # T intersection
            153: 'INTER',  # T intersection
            62: 'INTER',  # T intersection
            58: 'INTER',  # T intersection
            162: 'STOP',  # Stop sign
            169: 'STOP'   # Stop sign
        }
        self.led_colors = {
            0: 'WHITE',
            1: 'RED'
        }

        # apriltag detection filters
        self.decision_threshold = 10
        self.z_threshold = 0.5

        # PID Variables for driving
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

        # PID Variables for tailing a duckiebot
        self.distance = 0
        self.forward_P = 0.005
        self.forward_D = 0.001
        self.target = 0.40
        self.forward_error = 0
        self.last_fwd_err = 0
        self.last_distance = 0
        self.dist_margin = 0.05
        self.tailing = False

        # Service proxies
        # rospy.wait_for_service(f'/{self.veh}/led_emitter_node/set_pattern')
        # self.led_service = rospy.ServiceProxy(f'/{self.veh}/led_emitter_node/set_pattern', ChangePattern)

        self.loginfo("Initialized")
        # Shutdown hook
        rospy.on_shutdown(self.hook)
        
        # PID variables for drive_2
        self.P_v = 0.5
        self.D_v = 0.1
        self.P_omega = 0.015
        self.D_omega = 0.015
        
        self.last_P_v_err = 0
        self.last_P_omega_err = 0
        
        self.camera_center = 340
        self.turning_threshold = 160
        
        self.proportional = 0
        
        self.target_forward = 0.5
        self.current_forward = self.target_forward
        self.speed_up = 0.5
        
        self.target_lateral = 170
        self.current_lateral = 0
        
        self.leader_x = self.camera_center

    def run(self):
        if self.intersection_detected:
            self.intersection_sequence()
        elif self.detection:
            self.tailing = True
            self.tailPID()
        else:
            self.tailing = False
            self.drive()
            
    def run_2(self):
        rate = rospy.Rate(8)  # 8hz
        while not rospy.is_shutdown():
            if self.intersection_detected:
                self.intersection_sequence_2()

            self.drive_2()
            rate.sleep()
        

    def dist_callback(self, msg):
        self.distance = msg.data
        self.current_forward = msg.data

    def detection_callback(self, msg):
        self.detection = msg.data

    def centers_callback(self, msg):
        self.centers = msg.corners
        
        if self.detection and len(self.centers) > 0:
            # find the middle of the circle grid
            middle = 0
            i = 0
            while i < len(self.centers):
                middle += self.centers[i].x
                i += 1
            middle = middle / i
            
            # update the last known position of the bot ahead
            self.leader_x = middle
            

    def img_callback(self, msg):
        img = self.jpeg.decode(msg.data)
        self.intersection_detected = self.detect_intersection(msg)
        crop = img[:, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
                self.current_lateral = cx
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        # debugging
        if self.pub_img_bool:
            rect_img_msg = CompressedImage(
                format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

    def left_turn(self):
        rospy.loginfo("Beginning left turn")
        self.twist.v = self.velocity
        self.twist.omega = 3
        
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 1:
            self.vel_pub.publish(self.twist)

    def right_turn(self):
        rospy.loginfo("Beginning right turn")
        self.twist.v = self.velocity
        self.twist.omega = -4
        
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 0.5:
            self.vel_pub.publish(self.twist)

    def tailPID(self):
        # see how it behaves when the leader is turning at a curve

        # probably want to add intersection detection here (at the beginning)
        # but before that:
        # see how it behaves when the leader turns at an intersection

        # forward error is negative if duckiebot is too close, positive if too far
        self.forward_error = self.distance - self.target
        tail_P = self.forward_error * self.forward_P

        tail_d_error = (self.forward_error - self.last_fwd_err) / \
            (rospy.get_time() - self.last_time)
        self.last_fwd_err = self.forward_error
        self.last_time = rospy.get_time()
        tail_D = -tail_d_error * self.forward_D

        if self.forward_error < 0:
            # can change to slow down (or move back) instead of stopping
            self.twist.v = 0
            self.twist.omega = 0
        else:
            self.twist.v = self.velocity + tail_P + tail_D
            if self.proportional is None:
                self.twist.omega = 0
            else:
                # P Term
                P = -self.proportional * self.P

                # D Term
                d_error = (self.proportional - self.last_error) / \
                    (rospy.get_time() - self.last_time)
                self.last_error = self.proportional
                self.last_time = rospy.get_time()
                D = d_error * self.D
                self.twist.omega = P + D

        self.last_distance = self.distance
        self.vel_pub.publish(self.twist)

    def tail(self):
        # Might have been wonky because of bad target value, need to test
        if self.distance <= self.last_distance:
            # if the leader isn't moving, stop
            self.twist.v = 0
            self.twist.omega = 0
        elif self.distance > self.last_distance + self.dist_margin:
            # if the leader is moving, move
            if self.distance > self.target:
                self.twist.v = self.varying_velocity
                self.twist.omega = 0
                # if the duckiebot is lagging behind the leader, speed up
                # for this need to calculate the leader's relative velocity
                # if (self.distance - self.last_distance)/(self.last_time - rospy.get_time()) > 0
                # self.varying_velocity = self.varying_velocity + 0.01

        self.last_distance = self.distance
        self.last_time = rospy.get_time()
        self.vel_pub.publish(self.twist)

    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / \
                (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity
            self.twist.omega = P + D
            if self.pub_img_bool:
                self.loginfo("proportional: {}, P: {}, D: {}, omega: {}, v: {}".format(self.proportional, P, D, self.twist.omega, self.twist.v))

        self.vel_pub.publish(self.twist)

    def drive_2(self):
        
        # Correct omega based on distance to yellow line
        P_omega_error = self.target_lateral - self.current_lateral
        P_omega_correction = P_omega_error * self.P_omega

        D_omega_err = (P_omega_error - self.last_P_omega_err) / (rospy.get_time() - self.last_time)
        D_omega_correction = D_omega_err * self.D_omega

        if D_omega_correction > 2.5 or D_omega_correction < -2.5:
            # ignore derivative kicks
            D_omega_correction = 0
            	
        self.twist.omega = P_omega_correction + D_omega_correction
        
        # Correct velocity based on distance to bot ahead
        if not self.detection:
            self.current_forward = self.target_forward + self.speed_up
            
        P_v_error = self.current_forward - self.target_forward
        P_v_correction = P_v_error * self.P_v

        D_v_error = (P_v_error - self.last_P_v_err) / (rospy.get_time() - self.last_time)
        D_v_correction = D_v_error * self.D_v
        
        if P_v_correction < 0:
            # stop when bot too close
            self.twist.v = 0
            self.twist.omega = 0
        else:
            self.twist.v = P_v_correction + D_v_correction
        
        # Publish new velocity and Omega
        #rospy.loginfo(f"\ntarget_lateral: {self.target_lateral} \ncurrent_lateral: {self.current_lateral} \ntarget_foward: {self.target_forward} \ncurrent_forward: {self.current_forward}")
        #rospy.loginfo(f"\nP_v_correction: {P_v_correction} \nD_v_correction: {D_v_correction} \nP_omega_correction: {P_omega_correction} \nD_omega_correction: {D_omega_correction}")
        self.vel_pub.publish(self.twist)
        
        # update class atributes
        self.last_P_v_err = P_v_error
        self.last_P_omega_err = P_omega_error
        self.last_time = rospy.get_time()
    
    def intersection_sequence(self):
        # for now
        rospy.loginfo("detected intersection")
        # self.changeLED(1) ######## waiting for LED service takes too long

        # latency between detecting intersection and stopping
        wait_time = 1.5 # seconds
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + wait_time:
            self.drive()

        # stop
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)

        turn = 'STRAIGHT'
        wait_time = 5 # seconds
        start_time = rospy.get_time()
        last_x = 0
        straight_threshold = 100
        while rospy.get_time() < start_time + wait_time:
            # update last known x coordinate of the bot
            if self.detection and len(self.centers)>0:
                last_x = self.centers[0].x

        if last_x < 320 - straight_threshold:
            turn = 'LEFT'
        elif last_x > 320 + straight_threshold:
            turn = 'RIGHT'

        # self.changeLED(0)
        if self.tailing:
            # if the leader kept moving straight, move straight
            if self.detection:
                # edge case: really slow turning
                self.tailPID()
            # could check whether we are at stop or intersection
            # if the leader turned right, turn right
            elif turn == 'RIGHT':
                self.right_turn()
            # if the leader turned left, turn left
            elif turn == 'LEFT':
                self.left_turn()
        else:
            wait_time = 0.75 # seconds
            start_time = rospy.get_time()
            while rospy.get_time() < start_time + wait_time:
                self.twist.v = self.velocity
                self.twist.omega = 0
                self.vel_pub.publish(self.twist)

    def intersection_sequence_2(self):
    
        rospy.loginfo("Beginning intersection sequence")

        # continue driving until reaching stop line
        wait_time = 1.5 # seconds
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + wait_time:
            self.drive_2()

        # stop the bot
        self.twist.v = 0
        self.twist.omega = 0
        for i in range(8):
            self.vel_pub.publish(self.twist)

        # scan for bot to follow while waiting 3 seconds
        wait_time = 3 # seconds
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + wait_time:
            continue
        
        rospy.loginfo(f"Last detection: {self.leader_x}")

        if self.leader_x < self.camera_center - self.turning_threshold:
            self.left_turn()
        elif self.leader_x > self.camera_center + self.turning_threshold:
            self.right_turn()
        else:
            # drive straight
            wait_time = 0.75 # seconds
            start_time = rospy.get_time()
            while rospy.get_time() < start_time + wait_time:
                self.twist.v = self.velocity
                self.twist.omega = 0
                self.vel_pub.publish(self.twist)


    def detect_intersection(self, img_msg):
        # detect an intersection by finding the corresponding apriltags
        cv_image = None
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except CvBridgeError as e:
            self.log(e)
            return []

        # undistort the image
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (self.w, self.h), 0, (self.w, self.h))
        image_np = cv2.undistort(cv_image, self.K, self.D, None, newcameramtx)

        # convert the image to black and white
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # detect tags present in image
        tags = self.at_detector.detect(
            image_gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065)

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

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)

    def readYamlFile(self, fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         % (fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


if __name__ == "__main__":
    node = DuckiebotTailingNode("duckiebot_tailing_node")
    node.run_2()
