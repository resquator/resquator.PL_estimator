from flask import Flask
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


app = Flask(__name__)

@app.route("/")
def hello():
    filename = 'finalized_model.sav'
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    r=pd.DataFrame([['WLLLD', 'WDWWW', 4, 2, 15, 4, 2.87, 3.41, 2.92]])
    r.columns=['last_5_home','last_5_away','last_h_goals','last_a_goals',
            'last_wh_goals','last_wa_goals','odd_1','odd_N','odd_2']
    res = loaded_model.predict_proba(r)
    print(f'prediction is {np.round(res,3)*100}')
    return "Hello World! Daryl - model loaded..."



if __name__ == '__main__':
    app.run(debug=True)
