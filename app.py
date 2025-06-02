from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')
logging.info("Index page is running")
@app.route('/predictdata',methods=['GET','POST'])
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
        return render_template('home.html',results=results[0])
   

if __name__ == '__main__':
    app.run(host="127.0.0.1",port=1910,debug=True)