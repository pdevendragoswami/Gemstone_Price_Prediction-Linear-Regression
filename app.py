from flask import Flask,render_template,request,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods= ['GET','POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('form.html')

        else:
            data = CustomData(
                carat=float(request.form.get('carat')),
                depth = float(request.form.get('depth')),
                table = float(request.form.get('table')),
                x = float(request.form.get('x')),
                y = float(request.form.get('y')),
                z = float(request.form.get('z')),
                cut = request.form.get('cut'),
                color= request.form.get('color'),
                clarity = request.form.get('clarity'))

            final_data = data.get_data_as_dataframe()
            predict_pipeline_obj = PredictPipeline()
            pred_value = predict_pipeline_obj.predict_value(final_data)

            results = round(pred_value[0],2)

            return render_template('results.html',final_result = results)

    except  Exception as e:
        logging.info('There is some issue at predict_datapoint')
        raise CustomException(e,sys)



if __name__ == "__main__":
    app.run(host='0.0.0.0')

