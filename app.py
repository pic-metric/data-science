from flask import Flask, request, jsonify
from obj_detector import object_detection

def create_app():
    app = Flask(__name__)
    @app.route('/predictor', method = ['POST'])
    def predictor():
        """route receives an image url and id, returns image attributes"""

        # get info from backend
        lines = request.get_json(force=True)

        # get strings from json
        url = lines['url'] #backend will provide the key
        image_id = lines['image_id']

        # make sure input is correct
        assert isinstance(url, str)
        assert isinstance(image_id, int)

        #process image and generate prediction
        predictions = object_detection(url) #?????????????

        #send output to backend
        send_back =  {'image_id': image_id, 'predictions': predictions}

    return app
