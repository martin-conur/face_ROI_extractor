"""This python script locates faces in video or image and
stores it in a given folder (input in the bash line). The main
objective is to create face related datasets (ej: mask classifier)

It uses the dnn cv2 abilities, using a state of the art ssd face detector

ssd-resnet: 300x300 input"""
#libraries needed
import cv2
import numpy as np
import os
import time
import argparse

#command line ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="video", help="source of detection: video or image")
parser.add_argument("-i", "--input", default=0, help="path to the video or picture file")
parser.add_argument("-c", "--conf", default=0.5, help="confidence threshold for the face detector")
parser.add_argument("-s", "--save", default="default_dump", help="path to the dump folder")
args = vars(parser.parse_args())

try:
    os.mkdir(os.path.join(os.getcwd(), args["save"]))
except:
    pass

def face_detector(frame, face_detection_model):
    """This function it receives a frame and a localization model
       It returns the predictions and it boundig boxes coordinates"""
    #getting the height and width of the frame
    (h, w) = frame.shape[:2]
    #using cv2's blobFromImage to perform image transfomations required
    #to make a prediction by the face detection net (res10_300x300_ssd_iter_140000)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
    #pass the transformed image (blob) to the ssd net and obtain the detections
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()

    #empy lists to store the predictions (faces, locations and predictions)
    locations = []

    #for every detection
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2] #this is for this type of nnet output
        if confidence > float(args["conf"]):
            #get the bounding box coordinates (4 corners) and scales it to the img size
            (x0,y0,x1,y1) = (detections[0,0,i,3:7]*np.array([w,h,w,h])).astype(int)
            #we need to ensure that any point could be outside the pictures frame
            #so bottom left points should be greater than zero
            (x0, y0)= (max(0, x0), max(0, y0))
            #top right points should be less than the max width or height - 1
            (x1, y1) = (min(w-1,x1), min(h-1, y1))

            #extrat frame's ROI
            face = frame[y0:y1, x0:x1]
            #save the ROI
            cv2.imwrite(os.path.join(args["save"], f"{time.time()}.png"), face)
            locations.append((x0,y0,x1,y1))
    return locations


print("[INFO] Cargando deep learning model: res10_300x300_ssd_iter_140000 ...")
prototxt_path = os.path.join(os.getcwd(), "face_detector_model/deploy.prototxt")
weights_path = os.path.join(os.getcwd(), "face_detector_model/res10_300x300_ssd_iter_140000.caffemodel")
face_detection_model = cv2.dnn.readNet(prototxt_path, weights_path)

#if video as --mode argument
if args["mode"] == "video":
    #start camera livestream
    print("[INFO] Comenzando lectura de video ...")
    video = cv2.VideoCapture(args["input"])
    #check if there's video
    if not video.isOpened():
        print ("[!Error] No hay video, verifique la fuente o ruta del video")
        exit(0)
    time.sleep(1)

    #loop over every frame
    while True:
        ret, frame = video.read()
        #in this part it could be resized to assure a certain max shape
        frame = cv2.resize(frame, (600,400))

        (locations) = face_detector(frame, face_detection_model)

        for (x0, y0, x1, y1) in locations:
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0, 255, 0), 2)

        cv2.imshow("Face detector", frame)

        key = cv2.waitKey(1) & 0xFF

        #break the videostream if "q" is pressed
        if key == ord("q"):
            break
    #finish up
    cv2.destroyAllWindows()
    video.stop()

if args["mode"] == "image":
    print("[INFO] Leyendo imagen")
    try:
        image = cv2.imread(os.path.join(os.getcwd(), args["input"]))
        image = cv2.resize(image, (600,400))

        (locations) = face_detector(image, face_detection_model)

        for (x0, y0, x1, y1) in locations:
            cv2.rectangle(image, (x0,y0), (x1,y1), (0, 255, 0), 2)

        cv2.imshow("Face detector", image)
        cv2.waitKey(0)

    except:
        print("[!Error] No se puede leer la imagen en la ruta {}".format(os.path.join(os.getcwd(), args["input"])))

    #finish up
    cv2.destroyAllWindows()
