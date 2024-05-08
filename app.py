from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = r'E:\IIT_Guwahati\Semester_02\DA_526_Image_processing_with_ML\Project_final_files\Brain_tumor2\templates'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MODEL_PATH = r'E:\IIT_Guwahati\Semester_02\DA_526_Image_processing_with_ML\Project_final_files\Brain_tumor2\Brain_model_best.h5'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_prediction(image_path):
    # Load the saved model
    model = load_model(MODEL_PATH)
    
    # Load the image and preprocess it
    img = load_img(image_path, color_mode='grayscale', target_size=(224, 224))
    img = np.array(img) / 255.
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction
    prediction = model.predict(img)
    prediction = prediction.argmax(axis=-1)
    
    # Return the prediction label
    if prediction == 1:
        return 'Healthy'
    else:
        return 'Affected'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('result.html', prediction='No file uploaded')
    
    file = request.files['file']
    
    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return render_template('result.html', prediction='Invalid file type')
    
    # Save the uploaded file to the UPLOAD_FOLDER directory
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Get the prediction for the uploaded image
    prediction = get_prediction(file_path)
    
    # Render the result with the prediction as an argument
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
