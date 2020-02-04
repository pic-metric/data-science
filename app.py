from decouple import config
from flask import Flask, request, jsonify
from .obj_detector import object_detection
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = config(DATABASE_URL)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db = SQLAlcehmy(app)

    @app.route('/predictor', method=['POST'])
    def predictor():
        """route receives an image url and id, returns image attributes"""

        # get info from backend
        lines = request.get_json(force=True)

        # get strings from json
        url = lines['url']  # backend will provide the key
        image_id = lines['image_id']

        # make sure input is correct
        assert isinstance(url, str)
        assert isinstance(image_id, int)

        # process image and generate prediction
        predictions = object_detection(url)  # ?????????????

        # send output to backend
        send_back = {'image_id': image_id, 'predictions': predictions}
        return jsonify(send_back)

    @app.rout('/predict_batch', method=['POST'])
    def predict_batch():
        predictions = []
        for image in batch:
            pass

    return app
