from obj_detector import object_detection
import cv2

inputs = ['Test_Input/1.bicycles-SDOT-flickr.jpg', 
        'Test_Input/20170609_212223_001.jpg',
        'Test_Input/20190501_132302.jpg',
        'Test_Input/20190622_141804.jpg',
        'Test_Input/20191022_160001.jpg'
        ]

outputs = ['Test_Output/1.bicycles-SDOT-flickr.jpg', 
        'Test_Output/20170609_212223_001.jpg',
        'Test_Output/20190501_132302.jpg',
        'Test_Output/20190622_141804.jpg',
        'Test_Output/20191022_160001.jpg'
        ]

for i in range(len(inputs)):
    results = object_detection(inputs[i])
    img = results['image']
    cv2.imwrite(outputs[i], img)
    print(results['object_count'])
