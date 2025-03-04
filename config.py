import os
import rules
import pandas as pd
from dotenv import load_dotenv
import boto3
from botocore.config import Config
import logging

logging.basicConfig(level = logging.INFO)
load_dotenv()

digitalocean_config = {
    "SPACE_NAME": os.getenv('SPACE_NAME'),
    "REGION": os.getenv('REGION'), 
    "ACCESS_KEY": os.getenv('ACCESS_KEY'),
    "SECRET_KEY": os.getenv('SECRET_KEY'),
    "MODEL_FOLDER": "models/"
}


def generate_presigned_url(object_name, expiration=3600):
    """Generate a pre-signed URL for downloading models securely"""
    session = boto3.session.Session()
    client = session.client(
        's3',
        region_name=digitalocean_config["REGION"],
        endpoint_url=f"https://{digitalocean_config['REGION']}.digitaloceanspaces.com",
        aws_access_key_id=digitalocean_config["ACCESS_KEY"],
        aws_secret_access_key=digitalocean_config["SECRET_KEY"],
        config=Config(signature_version="s3v4")
    )

    url = client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': digitalocean_config["SPACE_NAME"],
            'Key': object_name 
        },
        ExpiresIn=expiration
    )

    return url


config_options = {
      'Detectron2' : {
            'roadvision_identifier': "Road Bumps and Potholes Detectron2 model",
            'model': 'Detectron2',
            'is_nested_image': True,
            'image_param': 'image',
            'model_url': lambda: generate_presigned_url("models/roadvisionDetectron2.pth"),
            # --------- MODEL CONFIG -----------
            'model_config': "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            'model_threshold': 0.5,
            'model_weight': "roadvisionDetectron2.pth",
            "device": "cuda",
            # --------- INFO SHEET CONFIG -----------
            'info_sheet': 'roadvisionSheet.xlsx',
            'num_of_classes': len(list(set(pd.read_excel("roadvisionSheet.xlsx", sheet_name="classes")['vision']))),
            "classes": pd.read_excel("roadvisionSheet.xlsx", sheet_name="classes").to_dict()['vision'],
            # --------- IMAGE AND VISUALIZER CONFIG -----------
            'save_blurry': True,
            'save_image': False,
            'blurry_threshold': 150,
            "visualizer_width": 355,
            "visualizer_height": 500,
            # --------- BUSINESS RULES CONFIG -----------
            "detect_rules": rules.rule_roadvision,
            "fine_tune": False,
      },

      'YOLO11' : {
            'roadvision_identifier': "Road Bumps and Potholes YOLO11 model",
            'model': 'YOLO11',
            'is_nested_image': True,
            'image_param': 'image',
            'model_url': lambda: generate_presigned_url("models/roadvisionYOLO11.pt"),
            # --------- MODEL CONFIG -----------
            'model_config': " ",
            'model_threshold': 0.5,
            'model_weight': "roadvisionYOLO11.pt",
            "device": "cuda",
            # --------- INFO SHEET CONFIG -----------
            'info_sheet': 'roadvisionSheet.xlsx',
            'num_of_classes': len(list(set(pd.read_excel("roadvisionSheet.xlsx", sheet_name="classes")['vision']))),
            "classes": pd.read_excel("roadvisionSheet.xlsx", sheet_name="classes").to_dict()['vision'],
            # --------- IMAGE AND VISUALIZER CONFIG -----------
            'save_blurry': True,
            'save_image': False,
            'blurry_threshold': 150,
            "visualizer_width": 355,
            "visualizer_height": 500,
            # --------- BUSINESS RULES CONFIG -----------
            "detect_rules": rules.rule_roadvision,
            "fine_tune": False,
      }

}