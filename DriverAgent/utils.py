import cv2
import base64
import numpy as np

def convert_to_base64(image_data: np.ndarray):
    """
    Convert the given image data to base64 encoding.
    
    Parameters:
        image_data (np.ndarray): The image data to be converted.
    
    Returns:
        str: The base64 encoded image.
    """
    _, buffer = cv2.imencode('.png', image_data)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64