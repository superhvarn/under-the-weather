from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import quote_plus
import logging
from flask_cors import CORS

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
CORS(app)
logging.getLogger('flask_cors').level = logging.DEBUG