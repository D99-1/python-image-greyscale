import cv2
import requests
import numpy as np
from io import BytesIO
import os

def download_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return np.array(bytearray(response.content), dtype=np.uint8)
    else:
        print("Error: Could not download the image.")
        return None

def convert_to_greyscale_and_blur(image_data, blur_strength=5):
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    if image is None:
        print("Error: Could not open or find the image.")
        return None
    
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(grey_image, (blur_strength, blur_strength), 0)
    
    return blurred_image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

def main():
    url = input("Please enter the URL of the image: ")
    output_image_name = 'blurred_image.jpg'
    output_image_path = os.path.join(os.path.dirname(__file__), output_image_name)

    image_data = download_image_from_url(url)
    if image_data is not None:
        blurred_image = convert_to_greyscale_and_blur(image_data)
        if blurred_image is not None:
            save_image(blurred_image, output_image_path)

main()
