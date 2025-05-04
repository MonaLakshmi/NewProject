from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
from torchvision import models, transforms
from torch import nn
import torch
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries


from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'
os.makedirs('static', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ----------------------- DATABASE INIT -----------------------
def init_db():
    conn = sqlite3.connect('tmp/auth.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            doctor_name TEXT NOT NULL,
            hospital_name TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT,
            patient_age INTEGER,
            description TEXT,
            image_filename BLOB,
            messages TEXT,
            timestamp TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()
def processImage(img_path):
    messages = []

    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super(FeatureExtractor, self).__init__()
            self.features = model.features
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.classifier.in_features, 1)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pooling(x)
            x = self.flatten(x)
            features = x
            x = self.classifier(x)
            return x, features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("first_model.h5")
    base_model = models.densenet201(pretrained=False)
    model_1 = FeatureExtractor(base_model).to(device)
    model_1.load_state_dict(torch.load("second_model.pth", map_location=device))
    model_1.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = load_img(img_path, target_size=(224, 224), color_mode='rgb')
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_label = 1 if prediction[0][0] >= 0.5 else 0

    if predicted_label == 1:
        messages.append("The image Given was IVCM")
    else:
        messages.append("The image Given was Slit-Lamp")
        img_cv = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_transformed = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _ = model_1(img_transformed)
            prob = torch.sigmoid(output).item()
            prediction = 1 if prob >= 0.5 else 0
        
        img_tensor = transform(img_rgb).to(device)

        unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])],
            std=[1/s for s in [0.229, 0.224, 0.225]]
        )
        img_for_lime = unnormalize(img_tensor.cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
        img_for_lime = (img_for_lime * 255).astype(np.uint8)
        explainer = lime_image.LimeImageExplainer()
        def predict_fn(images_np):
           model_1.eval()
           images_tensor = torch.tensor(images_np).permute(0, 3, 1, 2).float()
           images_tensor = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])(images_tensor / 255.0)
           images_tensor = images_tensor.to(device)
           with torch.no_grad():
              outputs, _ = model_1(images_tensor)
              probs = torch.sigmoid(outputs).cpu().numpy()
           return np.concatenate([1 - probs, probs], axis=1)
        explanation = explainer.explain_instance(
           image=img_for_lime,
           classifier_fn=predict_fn,
           top_labels=1,
           hide_color=0,
           num_samples=1000
        )
        predicted_label = explanation.top_labels[0]
        lime_img, mask = explanation.get_image_and_mask(
            label=predicted_label,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        highlighted_pixels = np.sum(mask != 0)
        total_pixels = mask.size
        highlighted_percentage = (highlighted_pixels / total_pixels) * 100
        print(f"LIME detected region covers: {highlighted_percentage:.2f}% of the image.")


        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mark_boundaries(lime_img, mask))
        axes[1].set_title(f"LIME Explanation\nDetected: {highlighted_percentage:.2f}%")
        axes[1].axis("off")

        save_path = os.path.join('static', 'result_1.png')  
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


        if prediction == 1:
            messages.append("The image contains Fungal Keratitis")
        else:
            messages.append("The image does not contain Fungal Keratitis")

    return messages


# ----------------------- ROUTES -----------------------
@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('home.html', username=session['username'],
                               doctor_name=session['doctor_name'],
                               hospital_name=session['hospital_name'])
    return redirect(url_for('login'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    app.config['UPLOAD_FOLDER'] = 'uploads'  # This is where you set the upload folder


    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/submit_image_1',methods=['POST'])
def submit_image_1():
    if 'username' not in session:
        return redirect(url_for('login'))

    patient_name = request.form['patient_name']
    patient_age = request.form['patient_age']
    description = request.form['description']
    image = request.files['image']
    joined_messages=[]
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        img=cv2.imread(image_path)
        img=cv2.resize(img,(128,128))
        img=img/255.0
        img_input=np.expand_dims(img,axis=0)
        model=load_model('unet_model.h5')
        pred_mask=model.predict(img_input)[0]
        pred_mask_binary=(pred_mask>0.5).astype(np.uint8)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask_binary.squeeze(), cmap='gray')
        plt.axis('off')

        save_path = os.path.join('static', 'result.png') 
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        with open("static/result.png", "rb") as f:
            image_blob = f.read()
        conn = sqlite3.connect('tmp/auth.db')
        c = conn.cursor()
        c.execute('''INSERT INTO history (user_id, patient_name, patient_age, description, image_filename, messages, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (session['user_id'], patient_name, patient_age, description,image_blob , '', datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return render_template("result.html",result_image='result.png')
        
    else:
        flash("Image upload failed.", "danger")

    return redirect(url_for('cross_image'))
    

@app.route('/submit_image', methods=['POST'])
def submit_image():
    if 'username' not in session:
        return redirect(url_for('login'))

    patient_name = request.form['patient_name']
    patient_age = request.form['patient_age']
    description = request.form['description']
    image = request.files['image']

    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        messages = processImage(image_path)
        joined_messages = " | ".join(messages)
        with open("static/result_1.png", "rb") as f:
            image_blob = f.read()
        # Save history in DB
        conn = sqlite3.connect('tmp/auth.db')
        c = conn.cursor()
        c.execute('''INSERT INTO history (user_id, patient_name, patient_age, description, image_filename, messages, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (session['user_id'], patient_name, patient_age, description,image_blob , joined_messages, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        return render_template('image_status.html', messages=messages)
    else:
        flash("Image upload failed.", "danger")

    return redirect(url_for('cross_image'))


@app.route('/check_image')
def check_image():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('image_status.html')
@app.route('/cross_image')
def cross_image():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('check_status.html')


@app.route('/history')
def view_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('tmp/auth.db')
    c = conn.cursor()
    c.execute("SELECT patient_name, patient_age, description, image_filename, messages, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC", (session['user_id'],))
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        image_blob=row[3]
        image_base64=base64.b64encode(image_blob).decode('utf-8')
        image_data_uri=f"data:image/png;base64,{image_base64}"
        history.append({
            "patient_name": row[0],
            "patient_age": row[1],
            "description": row[2],
            "image_filename": image_data_uri,
            "messages": row[4].split(" | "),
            "timestamp": row[5]
        })

    return render_template('history.html', history=history)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        doctor_name = request.form['doctor_name']
        hospital_name = request.form['hospital_name']

        conn = sqlite3.connect('tmp/auth.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, doctor_name, hospital_name) VALUES (?, ?, ?, ?)",
                      (username, password, doctor_name, hospital_name))
            conn.commit()
            flash('Registration successful. Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.')
        finally:
            conn.close()
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']

        conn = sqlite3.connect('tmp/auth.db')
        c = conn.cursor()
        c.execute("SELECT id, password, doctor_name, hospital_name FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password_input):
            session['user_id'] = user[0]
            session['username'] = username
            session['doctor_name'] = user[2]
            session['hospital_name'] = user[3]
            return redirect(url_for('home'))

        flash('Invalid username or password.')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.')
    return redirect(url_for('login'))


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=10000, debug=False)

