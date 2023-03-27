import picar

class PiDriver:
    def __init__(self):
        picar.setup()

        # Debug mode
        self.debug = False

        # DC Motor setup
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.offset = 0

        # Servo motor setup
        self.pan_servo = picar.Servo.Servo(1)
        self.pan_servo.offset = -10
        self.tilt_servo = picar.Servo.Servo(2)
        self.tilt_servo.offset = 0

        # Car params
        self.params = { 'speed':0,
	     	            'steer':90 }

    def change_speed(self, speed):
        # Checking max speed
        if -100 <= speed <= 100:
            self.params['speed'] = speed

            # Checking dir
            if self.params['speed'] >= 0:
                self.back_wheels.backward()
            else: self.back_wheels.forward()

            # Changing actual car speed
            self.back_wheels.speed = abs(self.params['speed'])

        if self.debug: print(f"speed: {self.params['speed']}")

    def increment_speed(self, increment):
        self.change_speed(self.params['speed'] + increment)

    def change_steer(self, steer):
        if 50 <= steer <= 140:
            self.params['steer'] = steer

            # Changing actual car steer
            self.front_wheels.turn(self.params['steer'])

        if self.debug: print(f"steer: {self.params['steer']}")

    def increment_steer(self, increment):
        self.change_steer(self.params['steer'] + increment)

    def stop(self):
        self.params['speed'] = 0
        self.back_wheels.stop()