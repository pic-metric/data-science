"""
Predicts and counts objects in image files
"""

import cv2
import cvlib as cv 
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as T

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

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


def predict(img, threshold):
    """Prediction Function"""

    # Define Tensor transfomation for Pytorch
    transform = T.Compose([T.ToTensor()])

    # Transform image
    image = transform(img)

    # Get prediction from model
    pred = model([image])

    # Get prediction classes
    labels = list(pred[0]['labels'].numpy())
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in labels]

    # Get Prediction boxes
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().numpy())]  
    pred_score = list(pred[0]['scores'].detach().numpy())

    # Get indexes for predictions above the threshold
    # Get list of index with score greater than threshold.
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    # Remove the predictions below the threshold
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    # Get object counts
    obj_counts = {}
    for obj in set(pred_class):
        obj_counts[obj] = pred_class.count(obj)
   
    return pred_boxes, pred_class, obj_counts, pred_score[:pred_t+1]


def object_detection(img_ref, threshold=0.75, rect_th=3, text_size=1, text_th=3):
    """ 
    Main functions gets predictions and creates image.
    """
    #Query database to get image data
    img_str = query(f"SELECT pic FROM Pics WHERE id = {img_ref}") 

        # Open image from sting 
    img = Image.open(BytesIO(img_str))   

    # Run prediction function to get predictions
    boxes, pred_class, object_count, pred_score = predict(img, threshold)
    
    # Convert image to use in OpenCV
    img = np.asarray(img) # Read image with cv2
    img = img[:, :, ::-1].copy() 
    
    # Run facial recognition if persons are found in picture
    if "person" in pred_class:
        faces, conf = cv.detect_face(img)
        object_count['faces'] = len(faces)
        
        for face in faces:
            x1 = face[0]
            y1 = face[1]
            x2 = face[2]
            y2 = face[3]
    
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    # annotate image with bounding boxes, class predictions, and prediction scores
    for i in range(len(boxes)):
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=(0,255,0), thickness=rect_th) 
        cv2.putText(image, pred_class[i] + " " + str(pred_score[i]), boxes[i][0],  
        cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th) 
    
    image = BytesIO(image)
    
    results = {}

    results['image'] = image
    results['object_count'] = object_count
    return results
 