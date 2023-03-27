import RPi.GPIO as GPIO
import time
import threading

class US_Sensor:
    def __init__(self, trigger_pin, echo_pin, colission_at=5):
        self.debug = False

        self.trigger = trigger_pin
        self.echo = echo_pin

        self.distance = 255
        self.colission = False
        self.colission_range = colission_at
        self.measuring = False

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)

        GPIO.add_event_detect(self.echo, GPIO.BOTH, callback=self.__irq_handler)

        #GPIO.add_event_detect(self.echo, GPIO.FALLING, callback=self.__falling_handler)
        #GPIO.add_event_detect(self.echo, GPIO.RISING, callback=self.__rising_handler)

    def start(self):
        while True:
            if self.debug: print("Sending new pulse")
            self.trigger_pulse()
            time.sleep(0.050)

    def trigger_pulse(self, pulse_len=0.00001):
        if self.debug: print("sending pulse")

        GPIO.output(self.trigger, True)
        time.sleep(pulse_len)
        GPIO.output(self.trigger, False)

        if self.debug: print("pulse sent")

    def __irq_handler(self, channel):
        if GPIO.input(channel):
            if self.debug: print("looks like it's rising")
            self.measuring = True
            self.start_time = time.time()

        elif not GPIO.input(channel) and self.measuring:
            if self.debug: print("looks like it's falling")

            self.stop_time = time.time()
            if self.debug: print(self.start_time, self.stop_time)

            pulse_duration = self.stop_time - self.start_time
            self.distance = (343*100*pulse_duration)/2
            self.measuring = False

            if self.debug: print(self.distance)

    def __rising_handler(self, channel):
        if self.debug: print("rising edge detected")
        
        # Stop detecting rising edge and wait for falling edge
        GPIO.remove_event_detect(self.echo)
        GPIO.add_event_detect(self.echo, GPIO.FALLING, callback=self.__falling_handler)
        
        self.start_time = time.time()
        if self.debug: print("we got here")

    def __falling_handler(self, channel):
        if self.debug: print("falling edge detected")
        
        # Stop detecting falling edge
        GPIO.remove_event_detect(self.echo)
        
        self.stop_time = time.time()
        pulse_duration = self.stop_time - self.start_time
        self.distance = (343*100*pulse_duration)/2
        print(self.distance)
        
        self.trigger_pulse()

    def measure(self):
        # Set trigger to high for 10 us
        GPIO.output(self.trigger, 1)
        time.sleep(0.00001)
        GPIO.output(self.trigger, 0)

        # Wait for echo pin to be high
        while not GPIO.input(self.echo):
            start_time = time.time()

        # Wait for echo pin to go low and stop counting
        while GPIO.input(self.echo):
            stop_time = time.time()

        # Calculating distance
        pulse_duration = stop_time - start_time
        distance = (340*100*pulse_duration)/2
        if self.debug: print(distance)
        
        return distance

if __name__ == "__main__":
    try:
        us_sensor = US_Sensor(5,6)
        #us_sensor.debug = True
        us_sensor.start()
        
    except KeyboardInterrupt:
         GPIO.cleanup()
