from http import client
import struct
import pickle
import socket
import sys
import os
import uuid
import string
import random
import paho.mqtt.client as mqtt
import skimage.exposure as exposure
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
flags.DEFINE_string('mqtt', '', 'mqtt communication to a broker')

def on_connect(client, userdata, flags, rc):
    if rc: print(f"A problem occured, could not connect to the broker: {FLAGS.mqtt}")
    else: print(f"Succesfully connected to broker: {FLAGS.mqtt}")

def on_disconnect(client, userdata, flags, rc=0):
    if rc: print(f"A problem occured, could not disconnect from the broker: {FLAGS.mqtt}")
    else: print(f"Succesfully disconnected from broker: {FLAGS.mqtt}")

def generate_client_id():
        # Making a client_id 23 characters long
        mac = str(hex(uuid.getnode()))[2:]
        random_str = "".join(random.choice(string.ascii_letters) for _ in range(11))
        return mac + random_str

def mqtt_connect(client_id, broker_address):
    client = mqtt.Client(client_id)
    client.connect(broker_address)
    return client

def send_mqtt_info(client, data):
    for topic, payload in data.items():
        client.publish(topic=topic, payload=payload, qos=1, retain=False) # publish

def get_mid_arr(arr: np.ndarray, k: int) -> np.ndarray:
    mid_arr = arr.copy()
    upper = np.triu_indices(mid_arr.shape[0], k=k)
    mid_arr[upper] = 0
    lower = np.tril_indices(mid_arr.shape[0], k=-k)
    mid_arr[lower] = 0
    return mid_arr

def get_upper_arr(arr, k: int) -> np.ndarray:
    upper_arr = arr.copy()
    lower_triangle_indices = np.tril_indices(upper_arr.shape[0], k= k - 1)
    upper_arr[lower_triangle_indices] = 0
    return upper_arr

def get_lower_arr(arr, k: int) -> np.ndarray:
    lower_arr = arr.copy()
    upper_triangle_indices = np.triu_indices(lower_arr.shape[0], k = - k + 1)
    lower_arr[upper_triangle_indices] = 0
    return lower_arr

def get_traffic_light_color(roi, hist_cutting_threshold, probability_boundary):
    # calculate 2D histograms for pairs of channels: GR
    hist = cv2.calcHist([roi], [1, 2], None, [256, 256], [0, 256, 0, 256])
    # hist is float and counts need to be scale to range 0 to 255
    scaled_hist = (exposure.rescale_intensity(hist, in_range=(0, 1), out_range=(0, 255)).clip(0, 255).astype(np.float64))

    # Split histogram into 3 regions
    (yellow_region, green_region, red_region) = (
        get_mid_arr(scaled_hist, hist_cutting_threshold),
        get_lower_arr(scaled_hist, hist_cutting_threshold),
        get_upper_arr(scaled_hist, hist_cutting_threshold),
    )

    # Count how many non zero values in each region
    (red_count, green_count, yellow_count) = (
        np.count_nonzero(red_region),
        np.count_nonzero(green_region),
        np.count_nonzero(yellow_region),
    )

    # Calculate total non-zero values
    total_count = red_count + green_count + yellow_count

    # Calculate red and green percentage
    red_percentage, green_percentage = (
        red_count / total_count,
        green_count / total_count,
    )

    # Logic for deciding color
    #print(f'red: {red_percentage}, green: {green_percentage}')
    if red_percentage > probability_boundary :
        predict = "red" 
    else: predict = "green" 

    #print(predict)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    cv2.imshow('traffic_light', roi)
    return predict, red_percentage, green_percentage


def check_proximity(frame, coords, boundary=0.3):
    _, _, _, y_max = tuple(map(int, coords))
    if y_max >= (frame.shape[0] - frame.shape[0]*boundary):
        return True
    return False

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    if FLAGS.mqtt:
        # Connect to a mqtt broker
        mqtt_info = dict()
        id = generate_client_id()
        client = mqtt_connect(id, FLAGS.mqtt)

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        #client.on_message = on_message
        client.loop_start()


    if FLAGS.video:
        #begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

    else:
        # Start streaming configs
        # Create socket for streaming
        HOST=''
        PORT=8485

        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        print('Socket created')
        s.bind((HOST,PORT))
        print('Socket bind complete')
        s.listen(10)
        print('Socket now listening')
        conn,addr=s.accept()

        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))

    out = None
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = 640
        height = 480
        fps = 10
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    stopped = False # Control flag for traffic light logic
    hist_thresh = 30
    probability_boundary = 0.09
    while True:
        if FLAGS.video:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_num += 1
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
        else:
            # Obtaining frame from stream socket
            while len(data) < payload_size:
                #print("Recv: {}".format(len(data)))
                data += conn.recv(4096)

            #print("Done Recv: {}".format(len(data)))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            #print("msg_size: {}".format(msg_size))
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            # Decoding the image into workable frame
            frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        frame_num += 1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        #class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        allowed_classes = ['person', 'car', 'bus', 'truck', 'traffic light']

        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass

        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image, object_data = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        else:
            image, object_data = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)

        
        if FLAGS.mqtt:
        # Checking to see which object were found
            mqtt_info['personDetected'] = False
            mqtt_info['carDetected'] = False
            mqtt_info['trafficLightColor'] = 'None'

            for object in object_data:
                if object[0] == 'person': 
                    mqtt_info['personDetected'] = str(True)
                    #if check_proximity(frame, object[2], 0.15): mqtt_info['smartStop'] = str(True)

                elif object[0] == 'traffic light': 
                    # Getting the frame region where the traffic light is
                    x_min, y_min, x_max, y_max = tuple(map(int, object[2]))
                    roi = frame[y_min:y_max, x_min:x_max, :]
                    # Getting the traffic light color
                    mqtt_info['trafficLightColor'], red_per, green_per = get_traffic_light_color(roi, hist_thresh, probability_boundary)
                    cv2.putText(img=frame, text=f"color: {mqtt_info['trafficLightColor']}", org=(0,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

                    # Printing values for calibration
                    if frame_num % 10 == 0: 
                        print(hist_thresh, probability_boundary)
                        print(f'red: {red_per}, green: {green_per}')
                        print(mqtt_info['trafficLightColor'])

                    # Sending smart stop signal
                    if mqtt_info['trafficLightColor'] in ['red'] and not stopped: 
                        stopped = True
                        client.publish(topic='smartStop', payload=str(stopped), qos=1, retain=False)
                    elif mqtt_info['trafficLightColor'] in ['green'] and  stopped: # If light is green and we're stopped
                        stopped = False
                        client.publish(topic='smartStop', payload=str(stopped), qos=1, retain=False)
                    
                elif object[0] in ['car', 'bus', 'truck']: 
                    mqtt_info['carDetected'] = str(True)
                    #if check_proximity(frame, object[2], 0.25): mqtt_info['smartStop'] = str(True)

            send_mqtt_info(client, mqtt_info)
        
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        if FLAGS.video: result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('a'): hist_thresh += 5
        elif key == ord('s'): hist_thresh -= 5
        elif key == ord('d'): probability_boundary += 0.01
        elif key == ord('f'): probability_boundary -= 0.01

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
