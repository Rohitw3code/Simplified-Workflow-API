from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# CORS(app)

f = "dataset.csv"
df = pd.read_csv(f)
dummy = df.copy()


@app.route('/api/',methods=['GET'])
def home():
    return jsonify({"message":["working","person"]})


@app.route("/api/df/datatypechange",methods=["POST","GET"])
def dataTypeChange():
    if request.method == 'GET':
        dtypes = {key: str(value) for key, value in dummy.dtypes.to_dict().items()}
        print(dtypes)
        return jsonify({'dtypes':dtypes})

    if request.method == 'POST':
        data = request.get_json()
        key = data.get('key')
        dtype = data.get('dtype').lower()
        if 'int' in dtype:
            dummy[key] = dummy[key].astype(int)
        elif 'float' in dtype:
            dummy[key] = dummy[key].astype(float)
        elif 'object' in dtype:
            dummy[key] = dummy[key].astype(str)
        else:
            return {'changed':False}
        dtypes = {key: str(value) for key, value in dummy.dtypes.to_dict().items()}
        return {'changed':True,'dtypes':dtypes}

@app.route("/api/df/missingdata/operation",methods=['POST','GET'])
def keyoperation():
    global dummy
    data = request.get_json()
    key = data.get('key')
    oprt = data.get('operation')

    if oprt == 'mean':
        dummy[key].fillna(dummy[key].mean(),inplace=True)
    if oprt == 'median':
        dummy[key].fillna(dummy[key].median(),inplace=True)
    if oprt == 'mode':
        mode = dict(dummy[key].mode())
        dummy[key].fillna(mode[0],inplace=True)
    if oprt == 'remove':
        dummy.dropna(subset=[key], inplace=True)
    if oprt == 'delete':
        dummy.drop(key,axis=1,inplace=True)
    if oprt == "replace":
        dummy[key].fillna(data.get("replace"),inplace=True)
        

    dtypes = {key: str(value) for key, value in dummy.dtypes.to_dict().items()}
    data = dummy.isna().sum()
    return jsonify({'updated':True,'data':data.to_dict(),'dtypes':dtypes})

@app.route('/api/df/<count>',methods=['GET'])
def dataframe(count:int = 5):
    data = df.dropna()
    data_val = data.head(int(count)).to_dict(orient='list')
    return {"data":data_val,"shape":data.shape}

@app.route('/api/df/dataencoding',methods=['GET','POST'])
def data_encoding():
    if request.method == 'GET':
        print(list(dummy.columns))
        return {"columns":list(dummy.columns)}
    

@app.route("/api/df/missingdata",methods=['GET'])
def missingData():
    global dummy
    dtypes = {key: str(value) for key, value in dummy.dtypes.to_dict().items()}
    return {"missing":dummy.isna().sum().to_dict(),"dtypes":dtypes}


if __name__ == "__main__":
    app.run(debug=True,port=5001)