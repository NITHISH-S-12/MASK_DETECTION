# import the necessary library and packages
from keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def Detection_of_Mask(frame, faceNet, maskNet):
    
    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the image using cv2.dnn.blobFromImage
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    #input to the FaceNet model the face detections
    faceNet.setInput(blob)

    #It processes the input image data and produces the output.
    detections = faceNet.forward()
    print(detections.shape)

    # initialize three empty list that are our list of Face, their corresponding locations,
    # and the list of prediction results(Classification results) from our face mask network
    Face = []
    locations = [] 
    Predictions = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # Filter the detections which are all weak and not ensuring the confidence which is greater than 0.5e
        if confidence > 0.5:
            
            # compute the (x, y)-coordinates of the bounding box for detected face/object in the frame
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #he bounding box coordinates are adjusted to ensure they fall within the dimensions of the frame. 
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #  The extracted face is then preprocessed by converting the color channel ordering from BGR to RGB, resizing it to a specified size (224x224 in this case),
            #  and applying further preprocessing (such as mean subtraction) using functions like cv2.cvtColor, cv2.resize, img_to_array, and preprocess_input.
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            #The preprocessed face is added to the faces list, and the bounding box coordinates are added to the locs list.
            # These lists will store the extracted face images and their corresponding locations for further processing.
            Face.append(face)
            locations.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(Face) > 0:
        # for faster inference we'll make batch predictions on *all*
        # Face at the same time rather than one-by-one predictions
        # in the above `for` loop
        Face = np.array(Face, dtype="float32")
        Predictions = maskNet.predict(Face, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locations, Predictions)

# It takes the paths to two files as input: prototxtPath (which specifies the architecture of the model) and weightsPath (which contains the learned weights of the model). 
# The function reads these files and initializes the faceNet model for face detection.
prototxtPath = r"F:\Mask_Detection\Face-Mask-Detection-master\face_detector\deploy.prototxt"
weightsPath = r"F:\Mask_Detection\Face-Mask-Detection-master\face_detector\res10_300x300_ssd_iter_140000.caffemodel" 
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#The function reads the model file and initializes the maskNet model for face mask detection
maskNet = load_model("F:\Mask_Detection\Face-Mask-Detection-master\Detector_mask.model")

# initialize the video stream for detection
print("starting video stream...detection")
vs = VideoStream(src=0).start()
 
# loop over the frames from the video stream
while True:
    #The code captures a frame from the video stream using vs.read() 
    # and resizes it using imutils.resize() and to ensure that the frame has a maximum width of 400 pixels, which can be helpful for efficient processing and display.
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #The Detection_of_Mask() function is called to detect faces in the frame and determine whether each face is wearing a face mask or not.
    # The function takes the frame, faceNet (the face detector model), and maskNet (the face mask detector model) as inputs and returns the detected face locations and corresponding predictions.
    (locations, Predictions) = Detection_of_Mask(frame, faceNet, maskNet)

    # looping over the detected face locations and their corresponding locations
    for (box, pred) in zip(locations, Predictions):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        #The code determines the class label ("Mask" or "No Mask") based on the prediction values. 
        #A color is assigned based on the label, where green is used for "Mask" and red is used for "No Mask".

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # The code includes the probability in the label and formats it as a string.  
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #The label text is then displayed on the frame using cv2.putText(), and a rectangle is drawn around the face using cv2.rectangle().
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the  frame of the output
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `N` key was pressed, break from the loop
    if key == ord("N"):
        break

cv2.destroyAllWindows()
vs.stop()
