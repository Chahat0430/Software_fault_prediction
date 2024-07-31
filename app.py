# app.py (Flask app)
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
from train_model import train_model
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded CSV file
        csv_file = request.files.get('csv_file')

        if csv_file and allowed_file(csv_file.filename):
            # Get the selected algorithm and sampling method
            algorithm = request.form.get('algorithm')
            sampling_method = request.form.get('sampling')

            # Save the uploaded file
            filename = secure_filename(csv_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            csv_file.save(file_path)

            # Redirect to the result page
            return redirect(url_for('result', algorithm=algorithm, filename=filename, sampling=sampling_method))

        else:
            # Handle invalid file type or missing file
            return "Invalid file format. Only CSV files are allowed.", 400

    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    algorithm = request.args.get('algorithm')
    filename = request.args.get('filename')
    sampling_method = request.args.get('sampling')

    # Ensure the algorithm and filename are provided
    if not algorithm or not filename:
        return "Algorithm or filename not specified.", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return "File not found.", 404

    # Train the model and get the results
    try:
        accuracy, precision, recall, f1, auc = train_model(algorithm, file_path, sampling_method)
    except Exception as e:
        return f"An error occurred while processing the file: {e}", 500

    # Render the results template
    return render_template('results.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc=auc)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
