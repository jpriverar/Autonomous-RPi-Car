from piCamera import PiCamera
import cv2
import numpy as np

def position(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global cX, cY, clicked
        cX, cY = x, y
        clicked = True
        
def on_low_H_thresh_trackbar(val):
    global low_H
    low_H = val
    
def on_high_H_thresh_trackbar(val):
    global high_H
    high_H = val
    
def on_low_S_thresh_trackbar(val):
    global low_S
    low_S = val
    
def on_high_S_thresh_trackbar(val):
    global high_S
    high_S = val
    
def on_low_V_thresh_trackbar(val):
    global low_V
    low_V = val
    
def on_high_V_thresh_trackbar(val):
    global high_V
    high_V = val
    
# Creating a named frame
cv2.namedWindow(winname='Frame')
cv2.setMouseCallback('Frame', position)

# Defining initial values
low_H = low_S = low_V = cX = cY = 0
high_H = high_S = high_V = 255
clicked = False
        
# Creating trackbars for calibration
cv2.createTrackbar('Low H', 'Frame', low_H, 255, on_low_H_thresh_trackbar)
cv2.createTrackbar('High H', 'Frame', high_H, 255, on_high_H_thresh_trackbar)
cv2.createTrackbar('Low S', 'Frame', low_S, 255, on_low_S_thresh_trackbar)
cv2.createTrackbar('High S', 'Frame', high_S, 255, on_high_S_thresh_trackbar)
cv2.createTrackbar('Low V', 'Frame', low_V, 255, on_low_V_thresh_trackbar)
cv2.createTrackbar('High V', 'Frame', high_V, 255, on_high_V_thresh_trackbar)

# Opening camera and starting main loop
camera = PiCamera()
while camera.cap.isOpened():
    ret, frame =  camera.cap.read()

    # Changing to different color spaces
    #lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Masking with target values
    mask = cv2.inRange(hsv_frame, (low_H, low_S, low_V), (high_H, high_S, high_V))
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Finding edges/contours
    edges = cv2.Canny(masked,100,200)
    
    # Finding line segments
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=20,maxLineGap=1)
    #if lines is not None:
    #    for line in lines:
    #        x1,y1,x2,y2 = line[0]
    #        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #else: print('No lines detected')
    
    # Checking if user wants to exit ('q' or esc)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: break
    
    
    # Binding callback function and printing pixel values at event location
    if clicked:
        #print(f'R:{frame[cY, cX, 2]}, G:{frame[cY, cX, 1]}, B:{frame[cY, cX, 0]}')
        print(f'H:{frame[cY, cX, 0]}, S:{frame[cY, cX, 1]}, V:{frame[cY, cX, 2]}')
        #print(f'L:{lab_frame[cY, cX, 0]}, A:{lab_frame[cY, cX, 1]}, B:{lab_frame[cY, cX, 0]}')
        clicked = False
        
    # Assigning resulting frame
    result_frame = masked
        
    # Drawing line in the middle of the screen
    cv2.line(result_frame, (camera.frame_width//2, 0), (camera.frame_width//2, camera.frame_height), (255,255,255), 1)
        
    cv2.imshow('Frame', result_frame)
    cv2.imshow('Edges', edges)


camera.cap.release()
cv2.destroyAllWindows()
print('Done')