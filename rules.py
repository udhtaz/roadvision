import pandas as pd


def rule_Detectron2(channel_config, sure_classes, present_classes, 
                            img_output, full_output, *args):
    '''
    This function will:
        - Check if the key detections are present in the image
        - check the number of detections in the image

    Input: inference output , Dictionary with details about the detections present in the images
    Output: Boolean which is True when all stipulated conditions are met by the image
    '''
    info_sheet = channel_config['info_sheet']

    # -------------- COUNT RULE ---------------
    detect_count = 0
    # is_minimum_count = None

    for detect in img_output.keys():
        detect_count += img_output[detect]
    
    # -------------- GOOD EXECUTION RULE ---------------
    rule_info = {"num_of_detects": detect_count}

    # -------------- FINE TUNE RULE ---------------
    if channel_config['fine_tune']:
        pass

    else:
        return rule_info