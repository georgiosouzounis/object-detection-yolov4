#!/usr/bin/env python3

import cv2
import numpy as np


# typical values for score_threshold = 0.5 and for nms_threshold - 0.4

def object_detection_analysis_with_nms(test_img, class_labels, class_colors, obj_detections_in_layers, score_threshold, nms_threshold):

	# get the image dimensions  
	img_height = test_img.shape[0]
	img_width = test_img.shape[1]

	result = test_img.copy()

	# declare lists for the arguments of interest: classID, bbox info, detection confidences
	class_ids_list = []
	boxes_list = []
	confidences_list = []
	# loop over each output layer 
	for object_detections_in_single_layer in obj_detections_in_layers:
		# loop over the detections in each layer
		for object_detection in object_detections_in_single_layer:
			# get the confidence scores of all objects detected with the bounding box
			prediction_scores = object_detection[5:]
			# consider the highest score being associated with the winning class
			# get the class ID from the index of the highest score
			predicted_class_id = np.argmax(prediction_scores)
			# get the prediction confidence
			prediction_confidence = prediction_scores[predicted_class_id]

			# consider object detections with confidence score higher than threshold
			if prediction_confidence > score_threshold:
				# get the predicted label
				predicted_class_label = class_labels[predicted_class_id]
				# compute the bounding box cooridnates scaled for the input image
				bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
				(box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
				start_x_pt = max(0, int(box_center_x_pt - (box_width / 2)))
				start_y_pt = max(0, int(box_center_y_pt - (box_height / 2)))

				# update the 3 lists for nms processing
				# - confidence is needed as a float 
				# - the bbox info has the openCV Rect format
				class_ids_list.append(predicted_class_id)
				confidences_list.append(float(prediction_confidence))
				boxes_list.append([int(start_x_pt), int(start_y_pt), int(box_width), int(box_height)])

	# NMS for a set of overlapping bboxes returns the ID of the one with highest 
	# confidence score while suppressing all others (non maxima)
	# - score_threshold: a threshold used to filter boxes by score 
	# - nms_threshold: a threshold used in non maximum suppression. 

	winner_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, score_threshold, nms_threshold)

	# create a list of winner boxes
	winner_box_list = []

	for winner_id in winner_ids:
		max_class_id = winner_id[0]
		box = boxes_list[max_class_id]
		start_x_pt = box[0]
		start_y_pt = box[1]
		box_width = box[2]
		box_height = box[3]
		winner_box_list.append(box)

		#get the predicted class id and label
		predicted_class_id = class_ids_list[max_class_id]
		predicted_class_label = class_labels[predicted_class_id]
		prediction_confidence = confidences_list[max_class_id]

		#obtain the bounding box end co-oridnates
		end_x_pt = start_x_pt + box_width
		end_y_pt = start_y_pt + box_height

		#get a random mask color from the numpy array of colors
		box_color = class_colors[predicted_class_id]

		#convert the color numpy array as a list and apply to text and box
		box_color = [int(c) for c in box_color]

		# print the prediction in console
		predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
		print("predicted object {}".format(predicted_class_label))

		# draw rectangle and text in the image
		cv2.rectangle(result, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
		cv2.putText(result, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

	return result, winner_box_list


def object_detection_iou(iou_image, detection_box, gt_box):
	start_pt_x_box_a = detection_box[0]
	start_pt_y_box_a = detection_box[1]
	end_pt_x_box_a = detection_box[0] + detection_box[2]
	end_pt_y_box_a = detection_box[1] + detection_box[3]
	cv2.rectangle(iou_image, (start_pt_x_box_a, start_pt_y_box_a), (end_pt_x_box_a, end_pt_y_box_a), (0, 255, 0), 2)
	cv2.putText(iou_image, "predicted bbox", (start_pt_x_box_a, start_pt_y_box_a-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

	start_pt_x_box_b = gt_box[0]
	start_pt_y_box_b = gt_box[1]
	end_pt_x_box_b = gt_box[0] + gt_box[2]
	end_pt_y_box_b = gt_box[1] + gt_box[3]
	cv2.rectangle(iou_image, (start_pt_x_box_b, start_pt_y_box_b), (end_pt_x_box_b, end_pt_y_box_b), (0, 0, 255), 2)
	cv2.putText(iou_image, "ground truth bbox", (start_pt_x_box_b, start_pt_y_box_b-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(start_pt_x_box_a, start_pt_x_box_b)
	yA = max(start_pt_y_box_a, start_pt_y_box_b)
	xB = min(end_pt_x_box_a, end_pt_x_box_b)
	yB = min(end_pt_y_box_a, end_pt_y_box_b)

	# compute the area of intersection rectangle
	intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the areas of both rectangles  separately
	detArea = (end_pt_x_box_a - start_pt_x_box_a + 1) * (end_pt_y_box_a - start_pt_y_box_a + 1)
	gtArea = (end_pt_x_box_b - start_pt_x_box_b + 1) * (end_pt_y_box_b - start_pt_y_box_b + 1)
	unionArea = detArea + gtArea - intersection_area

	# compute the intersection over union 
	iou_value = intersection_area / float(unionArea)
	cv2.putText(iou_image, "IoU: {:.4f}".format(iou_value), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("iou = {:.4f}".format(iou_value))

	# return the intersection over union value
	return iou_image, iou_value

