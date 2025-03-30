from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model("animal.h5")

# Class labels
class_labels = ['bears', 'crows', 'elephants', 'racoons', 'rats']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image
        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize
        
        # Make prediction
        prediction = model.predict(x)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index]) * 100
        
        # Return result
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'image_path': filepath
        })
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)