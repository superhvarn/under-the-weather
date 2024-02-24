from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import logging
import requests
from DiseasePredictionModel import generate_heatmaps_for_state



app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Apply CORS to all routes under /api/*

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://user:under-the-sea@172.20.10.2:3306/user_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstName = db.Column(db.String(120), nullable=False)
    lastName = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Input Validation (basic example, consider more robust validation for production)
    if not all(key in data for key in ('firstName', 'lastName', 'email', 'password')):
        return jsonify({'message': 'Missing data'}), 400

    # Check if user already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email already registered'}), 400

    try:
        user = User(
            firstName=data['firstName'],
            lastName=data['lastName'],
            email=data['email']
        )
        user.set_password(data['password'])
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'User registered successfully!'}), 201
    except Exception as e:
        return jsonify({'message': 'Registration failed', 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    # Retrieve user record based on the provided email
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Check if the provided password matches the hashed password stored in the database
    if user.check_password(password):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid email or password'}), 401

@app.route('/api/state', methods=['POST'])
def state():
    data = request.get_json()

    selectedState = data.get('selectedState')

    try:
        # Make a POST request to the /generate_heatmaps endpoint with the selected state
        response = generate_heatmaps_for_state(selectedState)
        if response:
            if response.status_code == 200:
                return jsonify({'message': 'Heatmaps generated successfully'}), 200
            else:
                return jsonify({'message': 'Failed to generate heatmaps'}), 500
        else:
            return jsonify({'message': 'No response received from heatmap generation function'}), 500
    except Exception as e:
        return jsonify({'message': 'Error processing request', 'error': str(e)}), 500


    

if __name__ == '__main__':
    app.run(debug=True)
