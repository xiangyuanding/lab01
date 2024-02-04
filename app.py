import cv2 as cv
import numpy as np
import gradio as gr

def generate_edge(image):
    """
    description: generates edges in the image
    parameters: image: numpy.ndarray
    returns: edges: numpy.ndarray
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 50, 150)
    dilated = cv.dilate(edges, (3,3), iterations=20)
    eroded = cv.erode(dilated, (3,3), iterations=10)
    edges = cv.bitwise_not(eroded)
    return edges

def detect_dark(image):
    """
    description: detects dark regions in the image
    parameters: image: numpy.ndarray
    returns: thresh: numpy.ndarray
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    return thresh

def generate_sky_mask(image):
    """
    description: generates a mask for the sky
    parameters: mask: numpy.ndarray, image_height: int
    returns: sky_mask: numpy.ndarray
    """
    height = image.shape[0]
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    white_board = np.zeros_like(image)
    for i in contours:
        _,y,_,_ = cv.boundingRect(i)
        if (y < height * 0.1):
            cv.drawContours(white_board, [i], -1, (255), thickness=cv.FILLED)
    return white_board

def run(image):
    """
    description: driver code, runs everything
    parameters: image: numpy.ndarray
    returns: image: numpy.ndarray
    """
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    edge_image = generate_edge(image)
    dark = detect_dark(image)
    edge_image = cv.bitwise_and(dark, edge_image)
    sky_mask = generate_sky_mask(edge_image)

    kernel = np.ones((35,35), np.uint8)
    sky_mask = cv.morphologyEx(sky_mask, cv.MORPH_OPEN, kernel)

    return sky_mask
if __name__ == "__main__":
    demo = gr.Interface(run, gr.Image(), "image",description="please upload an image with sky, and we will segment the sky for you.")
    demo.launch()
