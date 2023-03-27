import picar
import cv2

class PiCamera:
    def __init__(self, ):#input_camera=0):
        self.debug = False

        # Creating a camera object
        #self.cap = cv2.VideoCapture(input_camera)
	# Checks whether the camera was opened successfully
        #if not self.cap.isOpened():
        #    raise  RuntimeError("Unable to read camera feed")

        #self.frame_width = int(self.cap.get(3))
        #self.frame_height = int(self.cap.get(4))

        # Creating servo motors for camera direction
        self.pan_servo = picar.Servo.Servo(1)
        self.pan_servo.offset = -10
        self.tilt_servo = picar.Servo.Servo(2)
        self.tilt_servo.offset = 0

        # Camera parameters
        self.params = { 'pan': 85,
                        'tilt':90,
                        'recording': False }

    def change_pan(self, val):
        if 45 <= val <= 135:
            self.params['pan'] = val
            self.pan_servo.write(val)

        if self.debug: print(f"pan: {self.params['pan']}")

    def increment_pan(self, increment):
        self.change_pan(self.params['pan'] + increment)

    def change_tilt(self, val):
        if 70 <= val <= 135:
            self.params['tilt'] = val
            self.tilt_servo.write(val)

        if self.debug: print(f"tilt: {self.params['tilt']}")

    def increment_tilt(self, increment):
        self.change_tilt(self.params['tilt'] + increment)
