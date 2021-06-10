from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, render_template
import joblib as jb

app = Flask(__name__)
model = jb.load('kidney.joblib')

X = [[50.0, 90.0, 2.0, 1.02, 1.0, 1.0, 1.0, 0, 70.0,
      107.0, 7.2, 3.7, 12100.0, 1.0, 1.0, 0.0, 0.0, 1.0]]
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(X)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predictkidney():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have a kidney disease!"
    elif prediction == 0:
        pred = "You don't have a kidney disease."
    output = pred
    return render_template('index.html', predicted='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
