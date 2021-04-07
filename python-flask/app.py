from flask import Flask
import numpy as np
import pandas as pd
from flask import request, jsonify, render_template
import pickle
import requests,ssl
from flask_cors import CORS
app = Flask(__name__)
mdl = pickle.load(open('model.pkl', 'rb'))
CORS(app)
@app.route('/predict',methods=['POST'])
def predict():
    print(request)
    
    RI=float(request.json['ri'])


    Na=float(request.json['na'])

        
    Mg=float(request.json['mg'])

        
    Al=float(request.json['ai'])


    Si=float(request.json['si'])
        

    K=float(request.json['k'])
        

    Ca=float(request.json['ca'])
        

    Ba=float(request.json['ba'])

        
    Fe=float(request.json['fe'])

    datavalues=[[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]]
    data = pd.DataFrame(datavalues, columns = ['ri', 'na', 'mg','al','si', 'k', 'ca','ba','fe'])
    
    data[['ri','na','mg','ai','si','k','ca','ba','fe']] = StandardScaler().fit_transform(data[['ri','na','mg','ai','si','k','ca','ba','fe']])
    res=mdl.predict(data)
    return str(res[0])


        

if __name__=="__main__":
    app.run(debug=True)










    

