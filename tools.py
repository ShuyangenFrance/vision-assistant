import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from collections import Counter
from langchain.tools import tool
import numpy as np
import cv2
from sklearn.cluster import KMeans
from webcolors import CSS3_NAMES_TO_HEX, hex_to_rgb, rgb_to_name

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

tools=[]


@tools.append
@tool
def object_detection(image_path: str):
    """Use this tool to dectect objects of in the give image.
     It will return the detected objects and their quantities."""
    # TODO: find a way to search for the latest image
    threshold=0.5

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

        # Load and preprocess the input image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Filter predictions based on confidence threshold
    labels = predictions[0]['labels'][predictions[0]['scores'] > threshold]
    labels=[COCO_INSTANCE_CATEGORY_NAMES[label] for label in labels]
    labels = Counter(labels)
    # Generate the formatted string
    result = ', '.join(
        f"{count}: {label}s" if count > 1 else f"{count} {label}" for label, count in labels.items())

    return result

@tools.append
@tool
def cluster_colors(image_path):
    """Use this tool to find all colors in the image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Use K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Convert RGB values to color names
    color_names = []
    for color in dominant_colors:
        try:
            color_name = rgb_to_name((color[0], color[1], color[2]))
        except:
            # If the color name is not found, find the closest CSS3 color name
            color_name = find_closest_color_name(color)
        color_names.append(color_name)

    # Join the color names into a string, separated by commas
    color_names_string = ', '.join(color_names)

    return color_names_string





def find_closest_color_name(rgb_value):
    css3_hex_colors = CSS3_NAMES_TO_HEX

    # Convert the target RGB value to HEX
    target_hex = "#{:02x}{:02x}{:02x}".format(rgb_value[0], rgb_value[1], rgb_value[2])

    # Calculate the Euclidean distance between the target color and CSS3 colors
    distances = [np.linalg.norm(np.array(hex_to_rgb(css3_hex)) - np.array(rgb_value)) for css3_hex in css3_hex_colors.values()]

    # Find the index of the minimum distance
    min_distance_index = np.argmin(distances)

    # Get the closest CSS3 color name
    closest_color_name = list(css3_hex_colors.keys())[min_distance_index]

    return closest_color_name
