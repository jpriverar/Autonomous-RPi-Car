from mqtt_Client import MQTT_Client
from piDriver import PiDriver
from piCamera import PiCamera
from us_Sensor import US_Sensor
from detecting_lanes import laneDetector
import cv2
import threading
import time
import numpy as np
import socket
import struct
import pickle

class SmartCar:
    def __init__(self):
        # Initiating lane detector threshold values
        self.detector = laneDetector(2) # Degree of the polynomial fit
        self.detector.read_threshold_values('HSV')
        
        # Creating smart car components
        self.client = MQTT_Client()
        self.driver = PiDriver()
        self.camera = PiCamera()
        self.us_sensor = US_Sensor(5,6)
        
        # Creating the socket we're going to use to stream camera feed
        
        # Defining car params
        self.params = { 'help':False,
                        'debug':False,
                        'capture':False,
                        'auto':False,
                        'stream': False,
                        'mqtt_coms': True,
                        'base_speed': 23}

    def debug_mode(self):
        self.params['debug'] = True
        self.driver.debug = True
        self.camera.debug = True
        self.us_sensor.debug = False
        #self.lane_detector.debug = True

    def start(self):
        # Creating socket for video streaming
        if self.params['stream']:
            server_addr = ('192.168.0.104', 8485)
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(server_addr)
            connection = client_socket.makefile('wb')
            # Encoding params for video streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        
        # Initiating MQTT client as well as loop
        if self.params['mqtt_coms']:
            self.client.connect()
            self.client.loop_start()
            self.client.subscribe('smartStop')
        
        # Tilting camera down to see the lanes better
        self.camera.change_tilt(80)
        cap = cv2.VideoCapture(0)
        print('Frame dimentions: ', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Create deamon thread for reading ultrasonic_sensor
        threading.Thread(target=self.us_sensor.start, daemon=True).start()
        
        try:
            measurement_count = 0
            while(cap.isOpened()):
                # Checking there's no risk of colission
                dist = min(self.us_sensor.distance, 50)
                
                measurement_count += 1
                if measurement_count == 5:
                    self.client.publish(topic='PiCarDistance', payload=dist)
                    measurement_count = 0
                
                if dist < 6:
                    if not self.colission:
                        self.driver.stop()
                        self.colission = True
                elif dist > 10: self.colission = False
                    
                # Obtaining frame
                ret, frame = cap.read()
                
                # Sending encoded frame through socket
                if self.params['stream']:
                    try:
                        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                        stream_data = pickle.dumps(encoded_frame, 0)
                        stream_size = len(stream_data)
                        client_socket.sendall(struct.pack(">L", stream_size) + stream_data)
                    except Exception as e:
                        print('Could not stream frame')
                
                # Catching user input
                key = cv2.waitKey(1) & 0xFF

                # Checking to exit program (with 'q' or 'esc')
                if key == ord('q') or key == 27:
                    self.stop()
                    cap.release()
                    break

                # Otherwise execute user command
                self.__user_command(key)
                if self.params['help']: self.__show_help(frame)
                
                # If auto mode is active, show lane detection and threshold image
                if self.params['auto']:
                    # Start moving if there's no risk of colission and no smart stop signal is sent
                    if not self.colission and self.client.messages['smartStop'] != 'True':
                        self.driver.change_speed(self.params['base_speed'])
                    else: self.driver.stop()
                    
                    # Finding lanes in the frame
                    deviation, lanes, debug, gray = self.detector.find_reference(frame)
                    
                    # Correcting steering angle based on the reference
                    self.driver.change_steer(int((deviation+330)*0.14 + 50))
                    self.__sync_camera_steer()
                    
                    if self.params['debug']:
                        print(f'deviation: {deviation}')
                        print(f"steer: {self.driver.params['steer']}")
                
                    # Showing frame
                    cv2.imshow('Lane Detection', debug)
                    cv2.imshow('Gray', gray)
                # if auto mode is disabled only show the regular frame
                else:
                    cv2.imshow('Camera', frame)
                
                # Checking to save frame
                if self.params['capture']:
                    self.__save_frame('Assets', frame)
                    self.params['capture'] = False
                    
                # If the mqtt communication is enabled, then update:
                if self.params['mqtt_coms']:
                    self.send_mqtt_info()

        except KeyboardInterrupt:
            print("Interrupted by user")
            cap.release()
            self.stop()
            
    def send_mqtt_info(self, *args):
        self.client.publish(topic='ColissionRisk', payload=self.colission)
        self.client.publish(topic='PiCarSpeed', payload=self.driver.params['speed'])
        self.client.publish(topic='PiCarSteer', payload=self.driver.params['steer'])
        
        for (topic, payload) in args:
            self.client.publish(topic=topic, payload=payload)
            
    def stop(self):
        # Stopping motors and sending final msg
        self.__on_x_press()
        
        # Disconnecting from broker if previous connection was stablished
        if self.params['mqtt_coms']:
            self.client.loop_stop()
            self.client.disconnect()
        
        # Destroying remaining windows
        cv2.destroyAllWindows()
            
    def enable_streaming(self):
        self.params['stream'] = True
        
    def __sync_camera_steer(self):
        # Find out the value of the steering angle (50-140)
        steer_angle = self.driver.params['steer']
        
        # Adjust the camera pan accordingly (135-45)
        if steer_angle == 90:
            camera_angle = 85
        elif steer_angle < 90:
            camera_angle = 135 - (steer_angle-50)*1.2
        else:
            camera_angle = 85 - (steer_angle-90)*0.8
            
        self.camera.change_pan(camera_angle)
        
    def __save_frame(self, path, frame):
        name = '_'.join([str(x) for x in time.localtime()[0:6]])
        cv2.imwrite(f'{path}/{name}.jpg', frame)
        print('Frame saved')
        
    def __user_command(self, key):
        # Motor controls
        if key == ord('w'): self.__on_w_press()
        elif key == ord('s'): self.__on_s_press()
        elif key == ord('a'): self.__on_a_press()
        elif key == ord('d'): self.__on_d_press()
        elif key == ord('x'): self.__on_x_press()

        # Camera controls
        elif key == ord('i'): self.camera.increment_tilt(5)
        elif key == ord('k'): self.camera.increment_tilt(-5)
        elif key == ord('j'): self.camera.increment_pan(5)
        elif key == ord('l'): self.camera.increment_pan(-5)
        
        # Help and car data
        elif key == ord('h'): self.params['help'] ^= True
        
        # Save current frame
        elif key == 32: self.params['capture'] = True
        
        # Enable/Disable auto mode
        elif key == ord('m'):
            self.params['auto'] ^= True
            cv2.destroyAllWindows()
        
    def __on_w_press(self):
        if not self.colission:
            self.driver.increment_speed(5)
        
    def __on_s_press(self):
        self.driver.increment_speed(-5)
        
    def __on_a_press(self):
        self.driver.increment_steer(-10)
        self.__sync_camera_steer()
        
    def __on_d_press(self):
        self.driver.increment_steer(10)
        self.__sync_camera_steer()
        
    def __on_x_press(self):
        self.driver.stop()

    def __show_help(self, frame):
        text = ["help: h",
                "emergency break: x",
                "quit: q/esc",
                "speed up/down: w/s",
                "left/right: a/d",
                "camera up/down: i/k",
                "camera left/right: j/l",
                "save frame: space",
                "toggle auto mode: m"]

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,20)
        fontScale = 0.5
        fontColor = (0,0,255)
        lineType = 2
        linepos = 0

        for line in text:
            bottomLeftCornerOfText = (10,20+linepos*20)
            cv2.putText(frame, line, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            linepos = linepos + 1


if __name__ == "__main__":
    car = SmartCar()
    car.debug_mode()
    car.enable_streaming()
    car.start()

    
    
    
    
    
