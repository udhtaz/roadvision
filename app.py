# Imports: Flask and External calls
from flask import Flask, request, send_file
from flask_cors import CORS, logging
from werkzeug.utils import secure_filename
import requests
import wget
import aiohttp
import asyncio

# Imports: OS, IO and Environments
import io
import shutil
import sys
import os
import psutil
import gc
import nvidia_smi

# Imports: Image Libraries
from PIL import Image
import base64
import cv2
from dotenv import load_dotenv

# Imports: Detectron 2
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Imports: Data Manipulation
import numpy as np
import pandas as pd
import json
from random import randint
from flask_restx import Resource, Namespace, reqparse, Api
from werkzeug.datastructures import FileStorage

# Imports: Others
import logging
logging.basicConfig(level = logging.INFO)

# Imports: Custom Helpers
from helpers import roadvision_Helpers
import config 
from config import config_options


image_input = reqparse.RequestParser()
image_input.add_argument('returnType', default='img', type=str, location='form')
image_input.add_argument('selected_model', type=str, default='Detectron2', location='form')
image_input.add_argument('file', location='files', type=FileStorage)

heath_check_ns = Namespace('API Status', description='This is to check if the api is up and running...')
image_detection_ns = Namespace('Road Vision - Detection', description='This will generate rectangular bounding boxes or segmentation masks around the detections')

# Load Environment Variables
load_dotenv()  

# Setup Flask
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
port = int(os.environ.get("PORT", 80))
app.config["DEBUG"] = False # turn off in prod


api = Api(app, version='1.0', title='Road Vision', description='Road Vision API Documentation')

for roadvision_namespace in [heath_check_ns, image_detection_ns]:
    api.add_namespace(roadvision_namespace)

# Load Configurations
def load_models(roadvision_config):
    '''loads all available models'''
    # Create model object and define tracking parameters
    preds = {}
    try:
        for roadvision_channel, channel_config in roadvision_config.items():
            if str(channel_config["model_weight"]) not in preds:
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file(channel_config["model_config"]))
                cfg.MODEL.WEIGHTS = str(channel_config["model_weight"]) 
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = channel_config["num_of_classes"]
                if use_cuda:
                    cfg.MODEL.DEVICE = channel_config["device"] 
                else:
                    cfg.MODEL.DEVICE = "cpu" 
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = channel_config["model_threshold"]
                pred = DefaultPredictor(cfg)

                preds[str(channel_config["model_weight"])] = pred

        return preds
    except Exception as error:
        roadvision_notification = f"roadvision Error: {error}, Description: {roadvision_channel} model/config not found"
        logging.exception(roadvision_notification)

def get_channel_configurations(roadvision_channel, config_params):
    '''Returns a specific parameter from the config of a roadvision configuration'''
    roadvision_config = config.config_options

    try:
        channel_config = roadvision_config[roadvision_channel]
        try:
            roadvision_notification = channel_config[config_params]
        except Exception as error:
            roadvision_notification = f"{channel_config['roadvision_identifier']}: {error}, Description: {config_params} not found"
    except Exception as error:
        roadvision_notification = f"roadvision  Error: {error}, Description: {roadvision_channel} not found"
    
    return roadvision_notification

def get_image_file(attachment_id):
    ''' Downloads the specified image and returns a handle to the downloaded image'''
    filename = wget.download(attachment_id)

    return filename

def get_file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        file_size_byte = file_info.st_size
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if file_size_byte < 1024.0:
                return f"{file_size_byte:.1f}{x}"
            file_size_byte /= 1024.0

def get_picture_info(channel_config, pred, image, mask=False):
    ''' images the image and returns the following results
         image_quality, present_classes and coordinates
    '''
    picture_info = {
        'num_of_detects' : None,
        'image_quality' : None,
        'present_classes' : {},
        'coordinate': {},
        'masked_image': None,
    }

    roadvision_helper = roadvision_Helpers(channel_config)
    #get file size
    file_size = f"file size: {get_file_size(image)}"

    #get gpu usage if available
    gpu_resource_usage = ""
    if use_cuda:
        nvidia_smi.nvmlInit()
        device_count = nvidia_smi.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            inactive_mem = (mem.free/mem.total)*100
            active_mem = 100 - inactive_mem
            gpu_resource_usage += f"|Device {i}: {nvidia_smi.nvmlDeviceGetName(handle).decode()}| free_mem:{inactive_mem:.1f}%| in_use_mem:{active_mem:.1f}%|"
            
    else:
        gpu_resource_usage = "GPU not available"

    mem_info = psutil.virtual_memory()._asdict()
    active_mem = (mem_info["used"]/mem_info["total"])*100
    inactive_mem = 100 - active_mem
    cpu_usage = psutil.cpu_percent()
    cpu_resource_usage = f"cpu: {cpu_usage}%, free_mem: {inactive_mem:.1f}%, in_use_mem: {active_mem:.1f}%"
        
    
    roadvision_notification = f"{channel_config['roadvision_identifier']} Notification: roadvision Helper created."
    logging.info(roadvision_notification)
    logging.info(file_size)
    logging.info(cpu_resource_usage)
    logging.info(gpu_resource_usage)

    image_quality = roadvision_helper.check_blurry(image)
    if  image_quality == "Blurry":
        if channel_config['save_blurry']: pass
    else:
        image_quality = "Good"
        if channel_config['save_image']: pass
    
    if mask:
        img_output, present_classes, full_output, detected, sure_classes, masked_base64_str = roadvision_helper.inference_draw(pred, image)
        picture_info['masked_image'] = masked_base64_str
    else:
        img_output, present_classes, full_output, detected, sure_classes = roadvision_helper.inference(pred, image)
    

    coordinates = roadvision_helper.get_coordinates(full_output, detected)
    if channel_config['fine_tune']:
        rule_info, img_output =  channel_config['detect_rules'](channel_config, sure_classes, present_classes, 
                                                            img_output, full_output)
    else:
        rule_info = channel_config['detect_rules'](channel_config, sure_classes, present_classes, 
                                                            img_output, full_output)
    

    # if image_quality == "Blurry":
    #     picture_info['image_quality'] = image_quality
    # else:
    #     picture_info['image_quality'] = image_quality
    #     picture_info['present_classes'] =  img_output
    #     picture_info['coordinate'] = coordinates
    #     picture_info['num_of_detects'] = rule_info['num_of_detects']


    picture_info['image_quality'] = image_quality
    picture_info['present_classes'] =  img_output
    picture_info['coordinate'] = coordinates
    picture_info['num_of_detects'] = rule_info['num_of_detects']

    return picture_info


# ------------------------------- POSTMAN ROUTES -------------------------------
@app.route('/', methods=["GET"])
def health_check():
    """Confirms the service is running"""
    roadvision_notification = {"message" : "roadvision  is up and running..."}
    roadvision_notification["deployed_channels"] = {k: roadvision_config[k]['roadvision_identifier'] for k in roadvision_config}

    return roadvision_notification

@app.route('/<roadvision_channel>', methods=["GET"])
def health_check_channel(roadvision_channel):
    """Confirms the service is running"""
    try:
        roadvision_notification = f"{config_options[roadvision_channel]['roadvision_identifier']} roadvision is up and running..."
        logging.info(roadvision_notification)
    except Exception as error:
        roadvision_notification = "roadvision  Error:  Description: Channel not found"
        logging.exception(roadvision_notification)
    
    return roadvision_notification

@app.route('/analyse_bbox/<roadvision_channel>', methods=["POST"])
async def analyse_bbox(roadvision_channel):
    try:
        channel_config = roadvision_config[roadvision_channel]

        #extracting image from file
        return_type = request.form.to_dict(flat=False)['returnType'][0]
        uploaded_image = request.files['file']
        filename = secure_filename(uploaded_image.filename)
        uploaded_image.save(filename)
        image = filename

        picture_info = get_picture_info(channel_config, 
                                    roadvision_predictors[str(channel_config["model_weight"])], image)
        picture_info['present_classes'] = {str(key): value for key, value in picture_info['present_classes'].items()} 

        # analyse the picture
        try:
            roadvision_helper = roadvision_Helpers(channel_config)
            roadvision_notification = f"{channel_config['roadvision_identifier']} Notification: roadvision Helper created"
            logging.info(roadvision_notification)

            img_output, present_classes, full_output, \
            detected, sure_classes = roadvision_helper.inference(roadvision_predictors[str(channel_config["model_weight"])], 
                                                              image, pred_box_scale=False)
            pred_boxes = full_output["instances"].pred_boxes.tensor.cpu().numpy()
            pred_labels = full_output["instances"].pred_classes.cpu().numpy()
            pred_labels = [channel_config["classes"][label] for label in pred_labels]
            pred_confidence = full_output["instances"].scores.cpu().numpy()
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            detect_color = {detect:(randint(0,255), randint(0,255), randint(0,255)) for detect in pred_labels}

            pd_len = len(pred_labels)
            for i in range(pd_len):
                start_point = tuple(int(i.item()) for i in pred_boxes[i][:2])
                end_point = tuple(int(i.item()) for i in pred_boxes[i][2:]) # minus 10 so the text sits above the bounding box
                
                text_point = tuple(int(i.item())-10 for i in pred_boxes[i][:2])
                thickness = 2
                detect = pred_labels[i]
                box_color = detect_color[detect]
                bbox_txt = f"{detect} ({int(pred_confidence[i] * 100)}%)"
                cv2.rectangle(img, start_point, end_point, box_color, thickness, cv2.FILLED)
                cv2.putText(img, bbox_txt, text_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, thickness)

            cv2.imwrite('infered_image.jpg', img)

            os.remove(image)
            if return_type=="json":
                return picture_info
            return send_file('infered_image.jpg', mimetype='image/jpg')
        except Exception as error:
            roadvision_notification = f"roadvision Error: {error}, Description: error analysing {image}"
            logging.exception(roadvision_notification)
            return roadvision_notification
    except Exception as error:
        roadvision_notification = f"roadvision Error: {error}, Description: {roadvision_channel} not found"
        logging.exception(roadvision_notification)
        return roadvision_notification

@app.route('/analyse_segm/<roadvision_channel>', methods=["POST"])
async def analyse_segm(roadvision_channel):
    try:
        channel_config = roadvision_config[roadvision_channel]

        # Extracting image from file
        return_type = request.form.to_dict(flat=False)['returnType'][0]
        uploaded_image = request.files['file']
        filename = secure_filename(uploaded_image.filename)
        uploaded_image.save(filename)
        image = filename
        picture_info = get_picture_info(channel_config, 
                                    roadvision_predictors[str(channel_config["model_weight"])], image)
        
        picture_info['present_classes'] = {str(key): value for key, value in picture_info['present_classes'].items()}   
        # Analyse the picture
        try:
            roadvision_helper = roadvision_Helpers(channel_config)
            roadvision_notification = f"{channel_config['roadvision_identifier']} Notification: roadvision Helper created"
            logging.info(roadvision_notification)

            # Call inference_draw to get masked_base64_str
            img_output, present_classes, full_output, \
            detected, sure_classes, masked_base64_str = roadvision_helper.inference_draw(roadvision_predictors[str(channel_config["model_weight"])], 
                                                                                          image)

            # Save base64 image to file using the new method
            saved_image_path = roadvision_helper.save_base64_image(masked_base64_str, 'saved_image.jpg')

            os.remove(image)
            if return_type == "json":
                return picture_info
            return send_file(saved_image_path, mimetype='image/jpg')
        except Exception as error:
            roadvision_notification = f"roadvision Error: {error}, Description: error analysing {image}"
            logging.exception(roadvision_notification)
            return roadvision_notification
    except Exception as error:
        roadvision_notification = f"roadvision Error: {error}, Description: {roadvision_channel} not found"
        logging.exception(roadvision_notification)
        return roadvision_notification



# ------------------------------- SWAGGER FUNCTIONS AND ROUTES -------------------------------

# FUNCTIONS

async def image_bbox(roadvision_channel, return_type, uploaded_image):
    try:
        channel_config = roadvision_config[roadvision_channel]

        filename = secure_filename(uploaded_image.filename)
        uploaded_image.save(filename)
        image = filename

        picture_info = get_picture_info(channel_config, 
                                    roadvision_predictors[str(channel_config["model_weight"])], image)
        picture_info['present_classes'] = {str(key): value for key, value in picture_info['present_classes'].items()} 

        try:
            roadvision_helper = roadvision_Helpers(channel_config)
            roadvision_notification = f"{channel_config['roadvision_identifier']} Notification: roadvision Helper created"
            logging.info(roadvision_notification)

            img_output, present_classes, full_output, \
            detected, sure_classes = roadvision_helper.inference(roadvision_predictors[str(channel_config["model_weight"])], 
                                                              image, pred_box_scale=False)
            pred_boxes = full_output["instances"].pred_boxes.tensor.cpu().numpy()
            pred_labels = full_output["instances"].pred_classes.cpu().numpy()
            pred_labels = [channel_config["classes"][label] for label in pred_labels]
            pred_confidence = full_output["instances"].scores.cpu().numpy()
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            detect_color = {detect:(randint(0,255), randint(0,255), randint(0,255)) for detect in pred_labels}

            pd_len = len(pred_labels)
            for i in range(pd_len):
                start_point = tuple(int(i.item()) for i in pred_boxes[i][:2])
                end_point = tuple(int(i.item()) for i in pred_boxes[i][2:])             
                text_point = tuple(int(i.item())-10 for i in pred_boxes[i][:2]) # minus 10 so the text sits above the bounding box
                thickness = 2
                detect = pred_labels[i]
                box_color = detect_color[detect]
                bbox_txt = f"{detect} ({int(pred_confidence[i] * 100)}%)"
                cv2.rectangle(img, start_point, end_point, box_color, thickness, cv2.FILLED)
                cv2.putText(img, bbox_txt, text_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, thickness)

            cv2.imwrite('infered_image.jpeg', img)

            os.remove(image)
            if return_type=="json":
                return picture_info
            return send_file('infered_image.jpeg', mimetype='image/jpeg')

        except Exception as error:
            roadvision_notification = {"roadvision Error": f"{error}, Description: error analysing {image}"}
            logging.exception(roadvision_notification)
            return roadvision_notification
    except Exception as error:
        roadvision_notification = {"roadvision Error": f"{error}, Description: {roadvision_channel} not found"}
        logging.exception(roadvision_notification)
        return roadvision_notification

async def image_segm(roadvision_channel, return_type, uploaded_image):
    try:
        channel_config = roadvision_config[roadvision_channel]

        filename = secure_filename(uploaded_image.filename)
        uploaded_image.save(filename)
        image = filename
        picture_info = get_picture_info(channel_config, 
                                    roadvision_predictors[str(channel_config["model_weight"])], image)
        
        picture_info['present_classes'] = {str(key): value for key, value in picture_info['present_classes'].items()}   

        try:
            roadvision_helper = roadvision_Helpers(channel_config)
            roadvision_notification = f"{channel_config['roadvision_identifier']} Notification: roadvision Helper created"
            logging.info(roadvision_notification)

            # Call inference_draw to get masked_base64_str
            img_output, present_classes, full_output, \
            detected, sure_classes, masked_base64_str = roadvision_helper.inference_draw(roadvision_predictors[str(channel_config["model_weight"])], 
                                                                                          image)

            # Save base64 image to file using the new method
            saved_image_path = roadvision_helper.save_base64_image(masked_base64_str, 'saved_image.jpg')

            os.remove(image)
            if return_type == "json":
                return picture_info
                # return {"message": "Analysis completed successfully."}
            return send_file(saved_image_path, mimetype='image/jpg')
        except Exception as error:
            roadvision_notification = f"roadvision Error: {error}, Description: error analysing {image}"
            logging.exception(roadvision_notification)
            return roadvision_notification
    except Exception as error:
        roadvision_notification = f"roadvision Error: {error}, Description: {roadvision_channel} not found"
        logging.exception(roadvision_notification)
        return roadvision_notification


## ROUTES

@heath_check_ns.route('/')
class APIStatus(Resource):
    def get(self):
        """Confirms the service is running"""
        roadvision_notification = {"message" : "roadvision  is up and running..."}
        roadvision_notification["deployed_channels"] = {k: roadvision_config[k]['roadvision_identifier'] for k in roadvision_config}

        return roadvision_notification
    

@image_detection_ns.route('/bbox')
class ImageBoundingBox(Resource):   
    @image_detection_ns.expect(image_input)
    def post(self):
        args = image_input.parse_args()
        roadvision_channel = args['selected_model']
        return_type = args['returnType']
        uploaded_image = args['file'] 

        output = asyncio.run(image_bbox(roadvision_channel, return_type, uploaded_image))
        return output


@image_detection_ns.route('/segm')
class ImageSegmentation(Resource):   
    @image_detection_ns.expect(image_input)
    def post(self):
        args = image_input.parse_args()
        roadvision_channel = args['selected_model']
        return_type = args['returnType']
        uploaded_image = args['file']  

        output = asyncio.run(image_segm(roadvision_channel, return_type, uploaded_image))
        return output


if __name__ == '__main__':

    # Load Configs and Models
    use_cuda = torch.cuda.is_available()
    roadvision_config = config.config_options
    roadvision_predictors = load_models(roadvision_config)

    # log file
    setup_logger()

    asyncio.run(app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True))