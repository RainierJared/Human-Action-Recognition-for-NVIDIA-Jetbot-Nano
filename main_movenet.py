#Can optimize this by only running if human is detected
#Try out YOLOLight instead

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

camera = cv2.VideoCapture(0)

#Loading pre-trained model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

#elements in the array are as follows: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
EDGES ={
    (0,1): 'm',
    (0,2): 'c',
    (1,3): 'm',
    (2,4): 'c',
    (0,5): 'm',
    (0,6): 'c',
    (5,7): 'm',
    (7,9): 'm',
    (6,8): 'c',
    (8,10): 'c',
    (5,6): 'y',
    (5,11): 'm',
    (6,12): 'c',
    (11,12): 'y',
    (11,13): 'm',
    (13,15): 'm',
    (12,14): 'c',
    (14,16): 'c'
}

#Drawing keypoints
def draw_keypoints(input, keypoints, confidence):
    y,x,c = input.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        kx, ky, kc = kp
        if kc > confidence:
            cv2.circle(input, (int(ky), int(kx)), 4, (0,255,0), -1)

#Drawing connections/edges
def draw_connections(input, keypoints, edges, confidence):
    y,x,c = input.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2,x2,c2 = shaped[p2]
        
        if(c1 > confidence) & (c2 > confidence):
            cv2.line(input, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)
 
 #Loop through each person detected
def loop_people(input, keypoints, edges, confidence):
    for person in keypoints:
        draw_connections(input, person, edges, confidence)
        draw_keypoints(input, person, confidence)
 
#main function for video-camera
def main():
    while camera.isOpened():
        success, frame = camera.read()
        
        
        #Resize the image/frame
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,256)
        img_input = tf.cast(img, dtype=tf.int32)
        
        #Applying model
        out = movenet(img_input)
        keypoints = out['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        
        #Rendering points and connections
        loop_people(frame,keypoints,EDGES,0.25) 
        
        #Display camera
        cv2.imshow("Video Capture", frame)
        
        fps = camera.get(cv2.CAP_PROP_FPS)
        print(fps)  
        
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()
    
#Main function for image        
# def main():
#     #Loading image
#     img = cv2.imread('open_pose_project/test_image/image.jpg')

#     #Preprocessing the image
#     img_resized = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256) #expands the array at axis=0 and resizes to acceptable dimensions
#     img_input = tf.cast(img_resized, dtype=tf.int32) #casts to int32


#     #Run the model on the image
#     out = movenet(img_input)
#     keypoints = out['output_0'].numpy()[:,:,:51].reshape((6,17,3)) #appends to the new array y,x,confidence values from initial output
#     #print(keypoints)

#     #Render keypoints
#     loop_people(img, keypoints, EDGES, 0.25)
    
    
#     #Display image
#     cv2.imshow('Image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()