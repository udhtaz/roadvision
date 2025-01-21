# Imports: Flask and External calls
from flask import Flask, request, send_file
from flask_cors import CORS, logging
from werkzeug.utils import secure_filename
import wget
import aiohttp
import asyncio

# Imports: OS, IO and Environments
import io
import time
import datetime
import os
import psutil
import gc
import nvidia_smi

# Imports: Image Libraries
from PIL import Image
import cv2
from dotenv import load_dotenv

# Imports: Detectron 2, OpenVINO, and YOLO11
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from openvino.runtime import Core 
from ultralytics import YOLO

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

video_input = reqparse.RequestParser()
video_input.add_argument('returnType', default='vid', type=str, location='form')
video_input.add_argument('selected_model', type=str, default='Detectron2', location='form')
video_input.add_argument('file', location='files', type=FileStorage)

heath_check_ns = Namespace('API Status', description='This is to check if the api is up and running...')
image_detection_ns = Namespace('Road Vision - Image Detection', description='This will generate bounding boxes and segmentation masks around the image detections')
video_detection_ns = Namespace('Road Vision - Video Detection', description='This will generate bounding boxes and segmentation masks around the video detections')

# Load Environment Variables
load_dotenv()  

# Setup Flask
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'
port = int(os.environ.get("PORT", 80))
app.config["DEBUG"] = False # turn off in prod


api = Api(app, version='1.0', title='Road Vision', description='Road Vision API Documentation')

for roadvision_namespace in [heath_check_ns, image_detection_ns, video_detection_ns]:
    api.add_namespace(roadvision_namespace)

def load_models(roadvision_config):
    """Loads all available models."""
    preds = {}
    try:
        start = time.perf_counter()
        for roadvision_channel, channel_config in roadvision_config.items():
            if str(channel_config["model_weight"]) not in preds:
                model_type = channel_config["model"]
                
                if model_type == "Detectron2":
                    # Detectron2 Model
                    device = channel_config["device"] if use_cuda else "cpu"  # Determine device availability
                    cfg = get_cfg()
                    cfg.merge_from_file(model_zoo.get_config_file(channel_config["model_config"]))
                    cfg.MODEL.WEIGHTS = str(channel_config["model_weight"])
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = channel_config["num_of_classes"]
                    cfg.MODEL.DEVICE = device  # Use determined device
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = channel_config["model_threshold"]
                    pred = DefaultPredictor(cfg)
                
                elif model_type == "YOLO11":
                    # YOLOv11 Model
                    device = channel_config["device"] if use_cuda else "cpu"  # Determine device availability 
                    yolo_model = str(channel_config["model_weight"])
                    pred = YOLO(channel_config["model_weight"])
                    pred.overrides["conf"] = channel_config["model_threshold"]
                    pred.overrides["device"] = device  # Set device for YOLO
                    logging.info(f"yolo.checkpoint:[Checkpointer] Loading from {yolo_model} ...")
                    
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                preds[str(channel_config["model_weight"])] = pred
        
        elapsed = time.perf_counter() - start
        logging.info(f'Time taken to load all models: {elapsed:.6f} seconds')

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
    """
    Processes the image and returns key information including:
    image quality, detected classes, coordinates, and optionally a masked image.

    Args:
        channel_config (dict): Model configuration.
        pred: The model predictor (Detectron2 or YOLO11).
        image (str): Path to the input image.
        mask (bool): Whether to generate a masked image.

    Returns:
        dict: A dictionary containing detection results and metadata.
    """
    picture_info = {
        'num_of_detects': None,
        'image_quality': None,
        'present_classes': {},
        'coordinate': {},
        'masked_image': None,
    }

    roadvision_helper = roadvision_Helpers(channel_config)

    # Monitor file size
    file_size = f"file size: {get_file_size(image)}"

    # Monitor GPU usage if available
    gpu_resource_usage = ""
    if use_cuda:
        nvidia_smi.nvmlInit()
        device_count = nvidia_smi.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            inactive_mem = (mem.free / mem.total) * 100
            active_mem = 100 - inactive_mem
            gpu_resource_usage += (
                f"|Device {i}: {nvidia_smi.nvmlDeviceGetName(handle).decode()}| "
                f"free_mem:{inactive_mem:.1f}%| in_use_mem:{active_mem:.1f}%|"
            )
    else:
        gpu_resource_usage = "GPU not available"

    # Monitor CPU usage and memory
    mem_info = psutil.virtual_memory()._asdict()
    active_mem = (mem_info["used"] / mem_info["total"]) * 100
    inactive_mem = 100 - active_mem
    cpu_usage = psutil.cpu_percent()
    cpu_resource_usage = (
        f"cpu: {cpu_usage}%, free_mem: {inactive_mem:.1f}%, in_use_mem: {active_mem:.1f}%"
    )

    # Log system metrics
    roadvision_notification = f"{channel_config['roadvision_identifier']} Notification: roadvision Helper created."
    logging.info(roadvision_notification)
    logging.info(file_size)
    logging.info(cpu_resource_usage)
    logging.info(gpu_resource_usage)

    # Check image quality
    image_quality = roadvision_helper.check_blurry(image)
    if image_quality == "Blurry" and channel_config['save_blurry']:
        picture_info['image_quality'] = "Blurry"
    else:
        picture_info['image_quality'] = "Good"

    # Perform inference
    if isinstance(pred, YOLO):  # YOLO11 inference
        if mask:
            raise ValueError("Masked images are not supported for YOLO11.")
        img_output, present_classes, full_output, detected, sure_classes = roadvision_helper.inference(pred, image)
        coordinates = {}
        for result in full_output:
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = channel_config['classes'].get(cls_id, "Unknown")
                coordinates[cls_name] = coordinates.get(cls_name, [])
                coordinates[cls_name].append(box.xywh.tolist())
    else:  # Detectron2 inference
        if mask:
            img_output, present_classes, full_output, detected, sure_classes, masked_base64_str = roadvision_helper.inference_draw(pred, image)
            picture_info['masked_image'] = masked_base64_str
        else:
            img_output, present_classes, full_output, detected, sure_classes = roadvision_helper.inference(pred, image)
        coordinates = roadvision_helper.get_coordinates(full_output, detected)

    # Apply vision rules if necessary
    if channel_config['fine_tune']:
        rule_info, img_output = channel_config['detect_rules'](channel_config, sure_classes, present_classes, img_output, full_output)
    else:
        rule_info = channel_config['detect_rules'](channel_config, sure_classes, present_classes, img_output, full_output)

    # Update picture info
    picture_info.update({
        'present_classes': img_output,
        'coordinate': coordinates,
        'num_of_detects': rule_info.get('num_of_detects', 0),
    })

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


@app.route('/analyse_image_segm/<roadvision_channel>', methods=["POST"])
async def analyse_image_segm(roadvision_channel):
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

async def video_segm(roadvision_channel, return_type, uploaded_video=None):
    """
    Handles video segmentation for a given channel and return type.

    Args:
        roadvision_channel (str): The selected model channel.
        return_type (str): Either 'json' or 'vid'.
        uploaded_video: The uploaded video file.

    Returns:
        dict or str: JSON with detection counts or path to processed video.
    """
    try:
        # Get channel configuration
        channel_config = roadvision_config[roadvision_channel]
        predictor = roadvision_predictors[str(channel_config["model_weight"])]

        # Handle video source
        roadvision_helper = roadvision_Helpers(channel_config)
        if uploaded_video:
            video_source = roadvision_helper.save_uploaded_video(uploaded_video)
        else:
            video_source = 0  # Default to webcam if no file is provided

        # Determine output path if return_type is video
        output_path = None
        if return_type == "vid":
            output_path = "processed_video.mp4"

        # Process video based on the model type
        if channel_config["model"] == "YOLO11":
            result = roadvision_helper.yolo11_inference_video(predictor, video_source, return_type, output_path)
        else:
            result = roadvision_helper.process_video(predictor, video_source, return_type, output_path)

        # Cleanup uploaded video file
        if video_source != 0:
            os.remove(video_source)

        return result
    except Exception as error:
        logging.exception(f"Error processing video: {error}")
        return {"error": str(error)}, 500


## ROUTES

@heath_check_ns.route('/')
class APIStatus(Resource):
    def get(self):
        """Confirms the service is running"""
        roadvision_notification = {"message" : "roadvision  is up and running..."}
        roadvision_notification["deployed_channels"] = {k: roadvision_config[k]['roadvision_identifier'] for k in roadvision_config}

        return roadvision_notification
    
@image_detection_ns.route('/img_segm')
class ImageSegmentation(Resource):   
    @image_detection_ns.expect(image_input)
    def post(self):
        args = image_input.parse_args()
        roadvision_channel = args['selected_model']
        return_type = args['returnType']
        uploaded_image = args['file']  

        start_processing = time.perf_counter()
        output = asyncio.run(image_segm(roadvision_channel, return_type, uploaded_image))

        finished_processing = time.perf_counter() - start_processing
        logging.info(f'Time taken to process image with {roadvision_channel}: {finished_processing:.6f} seconds')

        return output

@video_detection_ns.route('/vid_segm')
class VideoSegmentation(Resource):
    @video_detection_ns.expect(video_input) 
    def post(self):
        args = video_input.parse_args()  
        roadvision_channel = args['selected_model']
        return_type = args['returnType']
        uploaded_video = args['file']

        start_processing = time.perf_counter()
        output = asyncio.run(video_segm(roadvision_channel, return_type, uploaded_video))
        finished_processing = time.perf_counter() - start_processing

        logging.info(f"Time taken to process video with {roadvision_channel}: {finished_processing:.6f} seconds")
        return output


if __name__ == '__main__':

    # Load Configs and Models
    use_cuda = torch.cuda.is_available()
    roadvision_config = config.config_options
    roadvision_predictors = load_models(roadvision_config)

    # log file
    setup_logger()

    asyncio.run(app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True))