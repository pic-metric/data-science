# from python-decouple import config
from flask import Flask, request, jsonify
from .obj_detector import object_detection
# from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)

    @app.route('/img_summary', methods=['GET'])
    def predictor():
        '''
        route receives an image/image id, returns processed image and
        a dict of attributes
        '''
        if request.method == 'GET':
            return 'this works'
    #
    # @app.rout('/batch_summary', method=['POST'])
    # def predict_batch():
    #     # we need to have a separate endpoint for batch uploads so that
    #     # we can work with as set of pictures, and return a set of pictures
    #     # with a combined summary.
    #     for image in batch:
    #         pass

    return app

    if __name__ == "__main__":
        app.run(debug=True, port=8080)
