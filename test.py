from obj_detector import object_detection
from query import get_image, put_image
import cv2

"""inputs = ['Test_Input/1.bicycles-SDOT-flickr.jpg',
        'Test_Input/20170609_212223_001.jpg',
        'Test_Input/20190501_132302.jpg',
        'Test_Input/20190622_141804.jpg',
        'Test_Input/20191022_160001.jpg'
        ]"""

"""outputs = ['Test_Output/1.bicycles-SDOT-flickr.jpg',
        'Test_Output/20170609_212223_001.jpg',
        'Test_Output/20190501_132302.jpg',
        'Test_Output/20190622_141804.jpg',
        'Test_Output/20191022_160001.jpg'
        ]"""

"""for i in range(len(inputs)):
    results = object_detection(inputs[i])
    img = results['image']
    cv2.imwrite(outputs[i], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(results['object_count'])"""

results = object_detection(img_ref=21)
img = results['image']
# cv2.imwrite('Test_Output/picid_6.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
put_image(image_id=21, processed_image=img, atts=results['object_count'])
print(img[:200], results['object_count'])
# print(get_image(5))
