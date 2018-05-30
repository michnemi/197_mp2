import csv
from detector import detector
from collections import namedtuple
import numpy as np
import cv2

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


preds = detector()
data = []

with open('labels.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	count = 0
	for row in reader:
		# print(row["filename"],[int(row["tl_x"]), int(row["tl_y"]), int(row["br_x"]), int(row["br_y"])], [preds[count][0][0], preds[count][0][1],preds[count][1][0],preds[count][1][1]])
		data.append(Detection(row["filename"],[int(row["tl_x"]), int(row["tl_y"]), int(row["br_x"]), int(row["br_y"])], [preds[count][0][0],preds[count][1][1],preds[count][1][0],preds[count][0][1]]))
		count += 1


avg_iou = 0
# loop over the example detections
for detection in data:
	# load the image
	image = cv2.imread("dataset/"+detection.image_path)

	# draw the ground-truth bounding box along with the predicted
	# bounding box
	cv2.rectangle(image, tuple(detection.gt[:2]), 
		tuple(detection.gt[2:]), (0, 255, 0), 2)
	cv2.rectangle(image, tuple(detection.pred[:2]), 
		tuple(detection.pred[2:]), (0, 0, 255), 2)

	# compute the intersection over union and display it
	iou = bb_intersection_over_union(detection.gt, detection.pred)
	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("{}: {:.4f}".format(detection.image_path, iou))
	avg_iou += iou

	# show the output image
	cv2.imshow("Image", image)
	cv2.imwrite("image.jpg", image)
	cv2.waitKey(0)

print("average IoU for dataset is: ", avg_iou/20)