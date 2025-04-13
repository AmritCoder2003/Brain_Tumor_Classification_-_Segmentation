import cloudinary
import cloudinary.uploader



from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
load_dotenv()


# Connect to MongoDB
mongourl = os.getenv("MONGO_URI")


client = MongoClient(mongourl)

db = client["brain_tumor_detection"]
doctors_collection = db["doctors"]
patients_collection = db["patients"]
classification_results = db["classification"]
segmentation_results = db["segmentation"]

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



cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUD_API_KEY"),
    api_secret=os.getenv("CLOUD_SECRET")
)


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


from jwt.exceptions import ExpiredSignatureError

@app.route('/dashboard', methods=['GET'])
def dashboard():
    token = session.get('jwt_token')
    if not token:
        flash("Please log in first", 'warning')
        return redirect(url_for('login'))

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except ExpiredSignatureError:
        flash("Session expired. Please log in again.", 'warning')
        session.clear()  
        return redirect(url_for('login'))
    except jwt.InvalidTokenError:
        flash("Invalid token. Please log in again.", 'danger')
        session.clear()
        return redirect(url_for('login'))

    doctor = doctors_collection.find_one({'email': payload['email']})
    if not doctor:
        flash("Doctor not found. Please log in again.", 'danger')
        session.clear()
        return redirect(url_for('login'))

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
        try:
            patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
        except (InvalidId, TypeError):
            patient = None
        if not patient:
            patient = patients_collection.find_one({'unique_id': patient_id})

        if not patient:
            flash("Patient not found.", "danger")
            return redirect(url_for('dashboard'))

        # Avoid shadowing collection names
        classification_data = list(classification_results.find({'patient_id': patient_id}))
        segmentation_data = list(segmentation_results.find({'patient_id': patient_id}))
        print(patient_id)
        print(classification_data)
        print(segmentation_data)

        return render_template(
            'view_patient.html',
            patient=patient,
            classification_results=classification_data,
            segmentation_results=segmentation_data
        )

    except Exception as e:
        print(f"Error while retrieving patient: {e}")  
        flash("Something went wrong while retrieving the patient.", "danger")
        return redirect(url_for('dashboard'))


@app.route('/delete_patient/<string:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    try:
        patients_collection.delete_one({'_id': ObjectId(patient_id)})
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

@app.route('/classification/<patient_id>', methods=['GET'])
def classification(patient_id):
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    return render_template('index.html', patient=patient)

@app.route('/segmentations/<patient_id>', methods=['GET'])
def segmentations(patient_id):
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    return render_template('segmentation.html', patient=patient)


    
@app.route('/logout')
def logout():
    session.pop('jwt_token', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))
        

@app.route('/predict/<patient_id>', methods=['POST'])
def predict(patient_id):
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    results = []
    files = request.files.getlist('images')

    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(filepath, folder="mri_scans/")
            cloudinary_url = upload_result['secure_url']

            # Load and preprocess the image
            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = classification_model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = float(np.max(predictions, axis=1)[0]) * 100

            result_text = "No Tumor" if class_labels[predicted_class_index] == 'notumor' else f"Tumor: {class_labels[predicted_class_index]}"

            result = {
                "patient_id": patient_id,
                "filename": filename,
                "cloudinary_url": cloudinary_url,
                "prediction": result_text,
                "confidence": round(confidence_score, 2)
            }

            classification_results.insert_one(result)
            results.append(result)

            # Optional: Remove local file after upload
            os.remove(filepath)

    return render_template('index.html', results=results, patient=patient)




@app.route('/segmentation/<patient_id>', methods=['GET', 'POST'])
def segmentation(patient_id):
    patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    segmented_results = []

    if request.method == 'POST':
        files = request.files.getlist('images')

        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Upload original to Cloudinary
                original_upload = cloudinary.uploader.upload(filepath, folder="mri_scans/")
                original_url = original_upload['secure_url']

                img = load_img(filepath, color_mode='grayscale', target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                mask = segmentation_model.predict(img_array)[0]
                mask = (mask.squeeze() * 255).astype(np.uint8)
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

                original_img = Image.open(filepath).convert('RGB')
                original_resized = original_img.resize((128, 128))
                original_array = np.array(original_resized)
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)

                if original_bgr.shape != mask_colored.shape:
                    mask_colored = cv2.resize(mask_colored, (original_bgr.shape[1], original_bgr.shape[0]))

                overlay = cv2.addWeighted(original_bgr, 0.7, mask_colored, 0.3, 0)

                # Save overlay temporarily
                segmented_filename = f"seg_{filename}"
                segmented_path = os.path.join(app.config['SEGMENTED_FOLDER'], segmented_filename)
                cv2.imwrite(segmented_path, overlay)

                # Upload segmented overlay to Cloudinary
                segmented_upload = cloudinary.uploader.upload(segmented_path, folder="segmented_scans/")
                segmented_url = segmented_upload['secure_url']

                result = {
                    "patient_id": patient_id,
                    "filename": filename,
                    "original_url": original_url,
                    "segmented_url": segmented_url
                }

                segmentation_results.insert_one(result)
                segmented_results.append(result)

                # Clean up local files
                os.remove(filepath)
                os.remove(segmented_path)

    return render_template('segmentation.html', results=segmented_results, patient=patient)




if __name__ == '__main__':
    app.run(debug=True)
