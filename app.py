from flask import Flask, request, render_template, send_from_directory, jsonify
import os

from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging
from src.database.db_handler import DatabaseHandler  

application = Flask(__name__)
app = application

db_handler = DatabaseHandler('predictions.db')  

@app.route('/')
def index():
    logging.info("Index page is running")
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    logging.info("Prediction page is running")
    if request.method == 'GET':
        logging.info("GET method is running")
        return render_template('home.html')
    else:
        logging.info("POST method is running")
        data = CustomData(
            Age=request.form.get('Age'),
            Sex=request.form.get('Sex'),
            Cholesterol=request.form.get('Cholesterol'),
            ChestPainType=request.form.get('ChestPainType'),
            RestingBP=request.form.get('RestingBP'),
            RestingECG=request.form.get('RestingECG'),
            FastingBS=request.form.get('FastingBS'),
            MaxHR=request.form.get('MaxHR'),
            ST_Slope=request.form.get('ST_Slope'),
            Oldpeak=request.form.get('Oldpeak'),
            ExerciseAngina=request.form.get('ExerciseAngina')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        Predict_Pipeline = PredictPipeline()
        results = Predict_Pipeline.predict(pred_df)
        logging.info("Prediction is done")
        
        # Save prediction to database only
        db_handler.save_prediction(pred_df, results[0])
            
        return render_template('home.html', results=results[0])

@app.route('/data')
def data_visualization():
    logging.info("Data visualization page is running")
    return render_template('Data.html')

@app.route('/download/notebook')
def download_notebook():
    notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebook')
    return send_from_directory(directory=notebook_path, path='EDA.ipynb', as_attachment=True)

@app.route('/predictions')
def view_predictions():
    """View all stored predictions"""
    try:
        predictions = db_handler.get_all_predictions()
        return render_template('predictions.html', predictions=predictions.to_dict('records'))
    except Exception as e:
        logging.error(f"Error retrieving predictions: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=1910, debug=True)
    
    db_handler.close()
