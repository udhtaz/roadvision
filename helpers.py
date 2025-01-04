import os
import cv2
import base64
import numpy as np
import pandas as pd
from detectron2.utils.visualizer import Visualizer
from datetime import datetime

class roadvision_Helpers:
    '''Containerized of helper functions for roadvision'''
    def __init__(self, channel_config):
        self.channel_config = channel_config

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

    def check_blurry(self, image):
        '''
        This function will:
            - check if image is blurry based on the set Laplacian threshold
        
        Input: Image
        Output: Blurry or Not Blurry

        '''

        image = cv2.imread(image)
        image = self.resize(image, height=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()

        if fm < self.channel_config['blurry_threshold']:
            return "Blurry"
        else:
            return "Not Blurry"

    def inference(self, model, image, pred_box_scale=True):
        '''
        This function will:
            - run inference on the images using the predictor object
            - analyze the output to identify the present classes, number of
            times they occur and other key data

        Input: Image
        Output: Results of inference on the image

        '''
        inputs = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if inputs.shape[2] > 3:
            # convert the image from RGBA2RGB
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGRA2BGR)

        full_output = model(inputs, )
        if pred_box_scale:
            full_output['instances'].pred_boxes.scale(
                self.channel_config["visualizer_width"]/inputs.shape[1], 
                self.channel_config["visualizer_height"]/inputs.shape[0])
        detected = full_output['instances'].pred_classes.tolist()
        pred_scores = full_output['instances'].scores.tolist()
        present_classes = [self.channel_config['classes'][i] for i in detected]

        sure_classes = []

        for i in range(len(present_classes)):
            if pred_scores[i] > 0.65:
                sure_classes.append(present_classes[i])

        img_output = {}
        for i in set(self.channel_config['classes']):
            img_output[i]= 0

        for i in set(present_classes):
            img_output[i] = present_classes.count(i)

        return img_output, present_classes, full_output, detected, sure_classes

    def inference_draw(self, model, image):
        '''
        This function will:
            - run inference on the images using the predictor object
            - analyze the output to idetify the present classes, number of
            times they occur and other key data.

        Input: Image
        Output: Results of inference on the image

        '''
        inputs = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if inputs.shape[2] > 3:
            # convert the image from RGBA2RGB
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGRA2BGR)

        full_output = model(inputs, )
        detected = full_output['instances'].pred_classes.tolist()
        v = Visualizer(inputs[:, :, ::-1], metadata={"thing_classes": list(self.channel_config['classes'].values())}, scale=1)
        out_mask = v.draw_instance_predictions(full_output["instances"].to("cpu"))
        masked_image = out_mask.get_image()

        full_output['instances'].pred_boxes.scale(
            self.channel_config["visualizer_width"]/inputs.shape[1], 
            self.channel_config["visualizer_height"]/inputs.shape[0])

        pred_scores = full_output['instances'].scores.tolist()
        present_classes = [self.channel_config['classes'][i] for i in detected]

        sure_classes = []

        for i in range(len(present_classes)):
            if pred_scores[i] > 0.65:
                sure_classes.append(present_classes[i])   

        img_output = {}
        for i in set(self.channel_config['classes']):
            img_output[i]= 0

        for i in set(present_classes):
            img_output[i] = present_classes.count(i)

        base64_str = cv2.imencode('.jpg',masked_image)[1].tostring()
        base64_str = base64.b64encode(base64_str)
        masked_base64_str = str("data:image/jpg;base64,")+str(base64_str.decode("utf-8"))

        return img_output, present_classes, full_output, detected, sure_classes, masked_base64_str
    
    def get_coordinates(self, full_output, detected):
        '''
        This function will:
            - analyze the output from the inference function and return coordinates of all classes detected
        Input: output from inference function
        Output: cooridinates of all detected classes

        '''
        coordinates = {}
        for i in set(detected):
            category_detect = full_output["instances"][full_output["instances"].pred_classes == i]
            coordinates[self.channel_config['classes'][i]] = category_detect.pred_boxes.get_centers().tolist()

        return coordinates

    def save_base64_image(self, base64_str, filename):
        base64_data = base64_str.split(",")[1]
        with open(filename, "wb") as file:
            file.write(base64.b64decode(base64_data))
        return filename

        '''Annotate a frame with bounding boxes and labels'''
        for bbox in analysis_data['bbox_data']:
            start_point = (bbox['left'], bbox['top'])
            end_point = (bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])
            color = (0, 255, 0)  # Green color for bounding box
            thickness = 2
            cv2.rectangle(frame, start_point, end_point, color, thickness)

            text = f"{bbox['label']} ({bbox['confidence']}%)"
            text_point = (bbox['left'], bbox['top'] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(frame, text, text_point, font, font_scale, color, thickness)