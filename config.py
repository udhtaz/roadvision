import rules
import pandas as pd
import datetime
import os


config_options = {
      'Detectron2' : {
            'roadvision_identifier': "Road Bumps and Potholes Detectron2 model",
            'model': 'Detectron2',
            'is_nested_image': True,
            'image_param': 'image',
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