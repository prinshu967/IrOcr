from flask import Flask
from .routes import ir_ocr

def create_app():
    app = Flask(__name__)
    app.register_blueprint(ir_ocr)
    return app
