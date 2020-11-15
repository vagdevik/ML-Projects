
import numpy as np
import cv2 as cv2
print("cv2 version:",cv2.__version__)
import os

import matplotlib.pyplot as plt



def fixColor(filepath):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_detected_image(image, filename):

	LABEL_FILE = "object_detection_classes_coco.txt"

	LABELS = open(LABEL_FILE).read().strip().split("\n")

	WEIGHTS="mask_rcnn_frozen_inference_graph.pb"
	MODEL_CONFIG="mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"


	basepath = os.path.dirname(__file__)

	label_file = os.path.join(basepath, LABEL_FILE)
	weights = os.path.join(basepath, WEIGHTS)
	model_config = os.path.join(basepath, MODEL_CONFIG)
	OUTPUT_PATH = os.path.join(basepath,'static','detections')
		
	np.random.seed(42) #Set seed so that we get the same results everytime
	LABELS = open(label_file).read().strip().split("\n")

	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

	net = cv2.dnn.readNetFromTensorflow(weights, model_config)

	img = cv2.imread(image)

	blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)

	net.setInput(blob)

	(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

	threshold=0.9

	detected_objects = []

	for i in range(0, boxes.shape[2]): #For each detection
	    classID = int(boxes[0, 0, i, 1]) #Class ID
	    confidence = boxes[0, 0, i, 2] #Confidence scores
	    if confidence > threshold:
	        (H, W) = img.shape[:2]
	        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H]) #Bounding box
	        (startX, startY, endX, endY) = box.astype("int")
	        boxW = endX - startX
	        boxH = endY - startY

	        # extract the pixel-wise segmentation for the object,       
	        mask = masks[i, classID]
	        # plt.imshow(mask)
	        # plt.show()
	        # print ("Shape of individual mask", mask.shape)
	        
	        # resize the mask such that it's the same dimensions of
	        # the bounding box, and interpolation gives individual pixel positions
	        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
	        # print ("Mask after resize", mask.shape)
	        # then finally threshold to create a *binary* mask
	        mask = (mask > threshold)
	        # print ("Mask after threshold", mask.shape)
	        
	        # extract the ROI of the image but *only* extracted the
	        # masked region of the ROI
	        roi = img[startY:endY, startX:endX][mask]
	        # print ("ROI Shape", roi.shape)
	        # grab the color used to visualize this particular class,
	        # then create a transparent overlay by blending the color
	        # with the ROI
	        color = COLORS[classID]
	        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

	        # Change the colors in the original to blended color
	        img[startY:endY, startX:endX][mask] = blended

	        color = COLORS[classID]
	        color = [int(c) for c in color]
	        # print (LABELS[classID], color)
	        detected_objects.append(LABELS[classID])
	        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
	        text = "{}: {:.4f}".format(LABELS[classID], confidence)
	        cv2.putText(img, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.imwrite(os.path.join(OUTPUT_PATH, filename), img)
	return ", ".join(detected_objects)
	