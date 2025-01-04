import rules
import pandas as pd
import datetime
import os


config_options = {
      'Detectron2' : {
            'roadvision_identifier': "Road Bumps and Potholes Detectron2 model",
            'is_nested_image': True,
            'image_param': 'image',
            # --------- MODEL CONFIG -----------
            'model_config': "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            'model_threshold': 0.5,
            'model_weight': "roadvision_detectron2.pth",
            "device": "cuda",
            # --------- INFO SHEET CONFIG -----------
            'info_sheet': 'roadvision_detectron2.xlsx',
            'num_of_classes': len(list(set(pd.read_excel("roadvision_detectron2.xlsx", sheet_name="classes")['vision']))),
            "classes": pd.read_excel("roadvision_detectron2.xlsx", sheet_name="classes").to_dict()['vision'],
            # --------- IMAGE AND VISUALIZER CONFIG -----------
            'save_blurry': True,
            'save_image': False,
            'blurry_threshold': 150,
            "visualizer_width": 355,
            "visualizer_height": 500,
            # --------- BUSINESS RULES CONFIG -----------
            "detect_rules": rules.rule_Detectron2,
            "fine_tune": False,
      },

      # 'OpenVINO' : {
      #       'roadvision_identifier': "Road Bumps and Potholes Open VINO model",
      #       'is_nested_image': True,
      #       'image_param': 'image',
      #       # --------- MODEL CONFIG -----------
      #       'model_config': " ",
      #       'model_threshold': 0.5,
      #       'model_weight': " ",
      #       "device": "cpu",
      #       # --------- INFO SHEET CONFIG -----------
      #       'info_sheet': ' ',
      #       'num_of_classes': len(list(set(pd.read_excel(" ", sheet_name="classes")[' ']))),
      #       "classes": pd.read_excel(" ", sheet_name="classes").to_dict()[' '],
      #       # --------- IMAGE AND VISUALIZER CONFIG -----------
      #       'save_blurry': True,
      #       'save_image': False,
      #       'blurry_threshold': 150,
      #       "visualizer_width": 355,
      #       "visualizer_height": 500,
      #       # --------- BUSINESS RULES CONFIG -----------
      #       "detect_rules": rules.rule_OpenVINO,
      #       "fine_tune": False,
      # },

      # 'YOLO_V10' : {
      #       'roadvision_identifier': "Road Bumps and Potholes YOLO V10 model",
      #       'is_nested_image': True,
      #       'image_param': 'image',
      #       # --------- MODEL CONFIG -----------
      #       'model_config': " ",
      #       'model_threshold': 0.5,
      #       'model_weight': " ",
      #       "device": "cuda",
      #       # --------- INFO SHEET CONFIG -----------
      #       'info_sheet': ' ',
      #       'num_of_classes': len(list(set(pd.read_excel(" ", sheet_name="classes")[' ']))),
      #       "classes": pd.read_excel(" ", sheet_name="classes").to_dict()[' '],
      #       # --------- IMAGE AND VISUALIZER CONFIG -----------
      #       'save_blurry': True,
      #       'save_image': False,
      #       'blurry_threshold': 150,
      #       "visualizer_width": 355,
      #       "visualizer_height": 500,
      #       # --------- BUSINESS RULES CONFIG -----------
      #       "detect_rules": rules.rule_YOLOv10,
      #       "fine_tune": False,
      # }

}