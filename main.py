from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
load_dotenv()
# Connect to MongoDB
mongourl = os.getenv("MONGO_URI")

client = MongoClient(mongourl)

db = client["brain_tumor_detection"]
predictions_collection = db["predictions"]
doctors_collection = db["doctors"]
patients_collection = db["patients"]

from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import jwt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

import cv2
from werkzeug.utils import secure_filename
from PIL import Image


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)
bycypt = Bcrypt(app)
from bson.objectid import ObjectId

JWT_SECRET = os.getenv("SECRET_KEY")
JWT_EXPIRY = int(os.getenv("JWT_EXPIRY_SECONDS") or 1800)


UPLOAD_FOLDER = 'static/uploads'
SEGMENTED_FOLDER = 'static/segmented'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER

# Load models
classification_model = load_model("models/model.h5")
segmentation_model = load_model("models/modelsegmentation.h5", compile=False)
class_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email'].lower()
        password = request.form['password']
        hashed_password = bycypt.generate_password_hash(password).decode('utf-8')

        if doctors_collection.find_one({'email': email}):
            flash('Email already exists. Please login.', 'danger')
            return redirect(url_for('login'))

        doctors_collection.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password,
            "patients": []
        })

        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form["password"]
        doctor = doctors_collection.find_one({'email': email})

        if doctor and bycypt.check_password_hash(doctor['password'], password):
            token = jwt.encode({
                'email': doctor["email"],
                'exp': datetime.now(timezone.utc) + timedelta(seconds=JWT_EXPIRY)
            },
                JWT_SECRET, algorithm='HS256'
            )
            session['jwt_token'] = token
            session['user_id'] = str(doctor['_id'])  # Optional: store doctor ID for later use
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/dashboard', methods=['GET'])
def dashboard():
    token = session.get('jwt_token')
    if not token:
        flash("Please log in first", 'warning')
        return redirect(url_for('login'))

    
    payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    doctor = doctors_collection.find_one({'email': payload['email']})
    
    # Fetch patients related to this doctor
    patients = list(patients_collection.find({'doctor_id': str(doctor['_id'])}))

    return render_template('dashboard.html', doctor=doctor, patients=patients)

    
   


    

from bson import ObjectId
import re


from uuid import uuid4

@app.route('/add_patient', methods=['POST'])
def add_patient():
    if 'user_id' not in session:
        flash("Unauthorized", "danger")
        return redirect(url_for('login'))

    name = request.form['name'].strip()
    age = request.form['age'].strip()
    gender = request.form['gender']
    email = request.form['email'].strip()
    phone = request.form['phone'].strip()
    address = request.form['address'].strip()
    symptoms = request.form['symptoms'].strip()
    medical_history = request.form['medical_history'].strip()
    notes = request.form['notes'].strip()

    # ✅ Backend Validation
    if not name or not age or not gender or not email or not phone:
        flash("All required fields must be filled out.", "danger")
        return redirect(url_for('dashboard'))

    if not age.isdigit() or int(age) <= 0:
        flash("Age must be a valid positive number.", "danger")
        return redirect(url_for('dashboard'))

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        flash("Invalid email address.", "danger")
        return redirect(url_for('dashboard'))

    if not re.match(r"^\d{10}$", phone):
        flash("Phone number must be 10 digits.", "danger")
        return redirect(url_for('dashboard'))

    # ✅ Generate Unique Patient ID
    unique_patient_id = str(uuid4())

    # ✅ Save patient data
    patient_data = {
        'doctor_id': session['user_id'],
        'unique_id': unique_patient_id,  # 🔑 Added field
        'name': name,
        'age': int(age),
        'gender': gender,
        'email': email,
        'phone': phone,
        'address': address,
        'symptoms': symptoms,
        'medical_history': medical_history,
        'notes': notes,
    }

    inserted_patient = patients_collection.insert_one(patient_data)
    patient_mongo_id = str(inserted_patient.inserted_id)

    doctor_collection = mongo.db.doctors
    doctor_collection.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$addToSet': {'patients': patient_mongo_id}}  # Avoid duplicates
    )

    flash("Patient added successfully.", "success")
    return redirect(url_for('dashboard'))


# Edit Patient
@app.route('/edit_patient/<patient_id>', methods=['GET', 'POST'])
def edit_patient(patient_id):
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})

    if request.method == 'POST':
        updated_data = {
            'name': request.form['name'],
            'age': request.form['age'],
            'gender': request.form['gender'],
            'email': request.form['email'],
            'phone': request.form['phone'],
            'address': request.form['address'],
            'symptoms': request.form['symptoms'],
            'medical_history': request.form['medical_history'],
            'notes': request.form['notes']
        }
        patients_collection.update_one({'_id': ObjectId(patient_id)}, {'$set': updated_data})
        return redirect(url_for('view_patient', patient_id=patient_id))

    return render_template('edit_patient.html', patient=patient)


from bson.objectid import ObjectId
from bson.errors import InvalidId

@app.route('/view_patient/<string:patient_id>')
def view_patient(patient_id):
    
    try:
        # First try ObjectId (Mongo's internal ID)
        try:
            patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
        except (InvalidId, TypeError):
            patient = None

        # Fallback: Try using unique_id (your custom UUID)
        if not patient:
            patient = patients_collection.find_one({'unique_id': patient_id})

        if not patient:
            flash("Patient not found.", "danger")
            return redirect(url_for('dashboard'))

        return render_template('view_patient.html', patient=patient)

    except Exception as e:
        print(f"Error while retrieving patient: {e}")  # helpful during dev
        flash("Something went wrong while retrieving the patient.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/delete_patient/<string:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    try:
        # Remove patient from the database
        patients_collection.delete_one({'_id': ObjectId(patient_id)})

        # Also remove this patient_id from the doctor's active_patients array
        doctor_id = session.get('doctor_id')
        if doctor_id:
            doctors_collection.update_one(
                {'_id': ObjectId(doctor_id)},
                {'$pull': {'patients': patient_id}}
            )

        flash("Patient deleted successfully.", "success")

    except Exception as e:
        print(f"Error deleting patient: {e}")
        flash("Something went wrong while deleting the patient.", "danger")

    return redirect(url_for('dashboard'))

@app.route('/classification')
def classification():
    return render_template('index.html')

@app.route('/segmentations')
def segmentations():
    return render_template('segmentation.html')


    
@app.route('/logout')
def logout():
    session.pop('jwt_token', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))
        


@app.route('/predict', methods=['POST'])
def predict():
    results = []
    files = request.files.getlist('images')

    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = classification_model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = float(np.max(predictions, axis=1)[0]) * 100

            result_text = "No Tumor" if class_labels[predicted_class_index] == 'notumor' else f"Tumor: {class_labels[predicted_class_index]}"

            results.append({
                "filename": filename,
                "prediction": result_text,
                "confidence": f"{confidence_score:.2f}"
            })

            predictions_collection.insert_one({
                "filename": filename,
                "prediction": result_text,
                "confidence": confidence_score,
                "segmented": None,  # Will be updated in segmentation route
                "timestamp": datetime.now(),
            })

    return render_template('index.html', results=results)



@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    segmented_results = []

    if request.method == 'POST':
        # Changed name from 'image' to 'images' to match HTML input
        files = request.files.getlist('images')

        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Load and preprocess the image
                img = load_img(filepath, color_mode='grayscale', target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict segmentation mask
                mask = segmentation_model.predict(img_array)[0]
                mask = (mask.squeeze() * 255).astype(np.uint8)

                # Apply color map
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

                # Load and resize original image
                original_img = Image.open(filepath).convert('RGB')
                original_resized = original_img.resize((128, 128))
                original_array = np.array(original_resized)
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)

                # Ensure shapes match before overlay
                if original_bgr.shape != mask_colored.shape:
                    mask_colored = cv2.resize(mask_colored, (original_bgr.shape[1], original_bgr.shape[0]))

                # Overlay prediction on image
                overlay = cv2.addWeighted(original_bgr, 0.7, mask_colored, 0.3, 0)

                # Save segmented image
                segmented_filename = f"seg_{filename}"
                segmented_path = os.path.join(app.config['SEGMENTED_FOLDER'], segmented_filename)
                cv2.imwrite(segmented_path, overlay)

                # Collect result to display
                segmented_results.append({
                    "filename": filename,
                    "segmented": segmented_filename
                })

                # Update in database (if used)
                predictions_collection.update_one(
                    {"filename": filename},
                    {"$set": {"segmented": segmented_filename}},
                    upsert=True  # Optional: to insert if not present
                )

    return render_template('segmentation.html', results=segmented_results)



if __name__ == '__main__':
    app.run(debug=True)
