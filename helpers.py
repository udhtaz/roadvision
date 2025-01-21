import os
import cv2
import base64
import numpy as np
import pandas as pd
from ultralytics import YOLO
from detectron2.utils.visualizer import Visualizer
from datetime import datetime
from werkzeug.utils import secure_filename

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
        """
        Runs inference on the given image using the specified model.
        Supports both Detectron2 and YOLO11 models.

        Args:
            model: The model (Detectron2 or YOLO11) to use for inference.
            image: Path to the image to process.
            pred_box_scale: Whether to scale prediction boxes (Detectron2 only).

        Returns:
            tuple: img_output, present_classes, full_output, detected_classes, sure_classes
        """
        inputs = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if inputs.shape[2] > 3:
            # Convert the image from RGBA to RGB
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGRA2BGR)

        if isinstance(model, YOLO):  # YOLO11 Inference
            results = model(inputs)  # Run YOLO11 inference
            detected_classes = []
            present_classes = []
            sure_classes = []
            img_output = {cls: 0 for cls in self.channel_config['classes'].values()}

            # Process YOLO results
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    cls_name = self.channel_config['classes'].get(cls_id, "Unknown")
                    detected_classes.append(cls_id)
                    present_classes.append(cls_name)
                    img_output[cls_name] += 1
                    if box.conf > 0.65:  # Confidence threshold for "sure" detections
                        sure_classes.append(cls_name)

            return img_output, present_classes, results, detected_classes, sure_classes

        else:  # Detectron2 Inference
            full_output = model(inputs)
            if pred_box_scale:
                full_output['instances'].pred_boxes.scale(
                    self.channel_config["visualizer_width"] / inputs.shape[1],
                    self.channel_config["visualizer_height"] / inputs.shape[0],
                )
            detected_classes = full_output['instances'].pred_classes.tolist()
            pred_scores = full_output['instances'].scores.tolist()
            present_classes = [self.channel_config['classes'][i] for i in detected_classes]

            sure_classes = []
            for i in range(len(present_classes)):
                if pred_scores[i] > 0.65:
                    sure_classes.append(present_classes[i])

            img_output = {cls: 0 for cls in self.channel_config['classes'].values()}
            for cls in present_classes:
                img_output[cls] += 1

            return img_output, present_classes, full_output, detected_classes, sure_classes

    def inference_draw(self, model, image):
        """
        Runs inference on the given image using the specified model
        and generates a visualized output with bounding boxes/masks.

        Args:
            model: The model (Detectron2 or YOLO11) to use for inference.
            image: Path to the image to process.

        Returns:
            tuple: img_output, present_classes, full_output, detected_classes, sure_classes, masked_base64_str
        """
        inputs = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if inputs.shape[2] > 3:
            # Convert the image from RGBA to RGB
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGRA2BGR)

        if isinstance(model, YOLO):  # YOLO11 Inference
            results = model(inputs)  # Run YOLO11 inference
            detected_classes = []
            present_classes = []
            sure_classes = []
            img_output = {cls: 0 for cls in self.channel_config['classes'].values()}

            # Process YOLO results
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    cls_name = self.channel_config['classes'].get(cls_id, "Unknown")
                    detected_classes.append(cls_id)
                    present_classes.append(cls_name)
                    img_output[cls_name] += 1
                    if box.conf > 0.65:  # Confidence threshold for "sure" detections
                        sure_classes.append(cls_name)

            # Generate annotated image
            annotated_image = results[0].plot()  # Use YOLO's plot method for visualization

            # Convert annotated image to Base64
            base64_str = cv2.imencode('.jpg', annotated_image)[1].tobytes()
            base64_str = base64.b64encode(base64_str)
            masked_base64_str = f"data:image/jpg;base64,{base64_str.decode('utf-8')}"

            return img_output, present_classes, results, detected_classes, sure_classes, masked_base64_str

        else:  # Detectron2 Inference
            full_output = model(inputs)
            detected_classes = full_output['instances'].pred_classes.tolist()
            pred_scores = full_output['instances'].scores.tolist()
            present_classes = [self.channel_config['classes'][i] for i in detected_classes]

            sure_classes = []
            for i in range(len(present_classes)):
                if pred_scores[i] > 0.65:
                    sure_classes.append(present_classes[i])

            # Generate annotated image using Detectron2 Visualizer
            v = Visualizer(
                inputs[:, :, ::-1],
                metadata={"thing_classes": list(self.channel_config['classes'].values())},
                scale=1,
            )
            out_mask = v.draw_instance_predictions(full_output["instances"].to("cpu"))
            masked_image = out_mask.get_image()

            # Convert annotated image to Base64
            base64_str = cv2.imencode('.jpg', masked_image)[1].tobytes()
            base64_str = base64.b64encode(base64_str)
            masked_base64_str = f"data:image/jpg;base64,{base64_str.decode('utf-8')}"

            img_output = {cls: 0 for cls in self.channel_config['classes'].values()}
            for cls in present_classes:
                img_output[cls] += 1

            return img_output, present_classes, full_output, detected_classes, sure_classes, masked_base64_str

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
    
    def process_video(self, predictor, video_source, return_type="json", output_path=None):
        """
        Processes a video or camera feed for object detection.

        Args:
            predictor: Loaded model predictor (e.g., Detectron2 or YOLO).
            video_source (str/int): Path to video file or camera index.
            return_type (str): 'json' to return JSON with class counts, 'video' to save and return video output.
            output_path (str): Path to save processed video (if return_type is 'video').

        Returns:
            dict: JSON with counts for each class (if return_type is 'json').
            str: Path to saved video (if return_type is 'video').
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise Exception(f"Error opening video source: {video_source}")

        # Initialize video writer if saving the output
        out = None
        if return_type == "vid" and output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0
        class_counts = {cls: 0 for cls in self.channel_config['classes'].values()}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            outputs = predictor(frame)
            detected_classes = outputs['instances'].pred_classes.tolist()
            for cls_id in detected_classes:
                cls_name = self.channel_config['classes'].get(cls_id, "Unknown")
                class_counts[cls_name] += 1

            # Visualization
            if return_type == "vid" and out:
                v = Visualizer(frame[:, :, ::-1], metadata={"thing_classes": list(self.channel_config['classes'].values())}, scale=1)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                processed_frame = v.get_image()[:, :, ::-1]
                out.write(processed_frame)

            frame_count += 1

        cap.release()
        if out:
            out.release()

        # If return type is JSON, return class counts
        if return_type == "json":
            return {
                "class_counts": class_counts,
                "frame_count": frame_count
            }
        elif return_type == "vid":
            return output_path

    def save_uploaded_video(self, uploaded_video):
        """
        Saves an uploaded video file locally and returns its file path.
        """
        filename = secure_filename(uploaded_video.filename)
        uploaded_video.save(filename)
        return filename

    def yolo11_inference_image(self, model, image_path):
        """
        Runs YOLO11 inference on a single image.

        Args:
            model: YOLO11 model instance.
            image_path (str): Path to the input image.

        Returns:
            dict: Dictionary containing class detections and their counts.
        """
        results = model(image_path)  # Run inference
        class_counts = self._process_yolo_results(results)
        return class_counts

    def yolo11_inference_video(self, model, video_source, return_type="json", output_path=None):
        """
        Runs YOLO11 inference on a video source.

        Args:
            model: YOLO11 model instance.
            video_source (str/int): Path to video file or camera index.
            return_type (str): 'json' to return class counts, 'video' to save the annotated video.
            output_path (str): Path to save the processed video if return_type is 'video'.

        Returns:
            dict or str: JSON with class counts (if return_type is 'json') or video path.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise Exception(f"Error opening video source: {video_source}")

        # Initialize video writer if saving output
        out = None
        if return_type == "vid" and output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        class_counts = {cls: 0 for cls in self.channel_config['classes'].values()}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO inference on the frame
            results = model(frame, stream=False)
            frame_counts = self._process_yolo_results(results)
            for cls, count in frame_counts.items():
                class_counts[cls] += count

            # Visualization for video output
            if return_type == "vid" and out:
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

        cap.release()
        if out:
            out.release()

        if return_type == "json":
            return {"class_counts": class_counts}
        elif return_type == "vid":
            return output_path

    def _process_yolo_results(self, results):
        """
        Processes YOLO11 results and returns class counts.

        Args:
            results: YOLO11 inference results.

        Returns:
            dict: Dictionary with counts of detected classes.
        """
        class_counts = {cls: 0 for cls in self.channel_config['classes'].values()}
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                cls_name = self.channel_config['classes'].get(class_id, "Unknown")
                class_counts[cls_name] += 1
        return class_counts