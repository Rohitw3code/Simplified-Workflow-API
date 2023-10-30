from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import RegressionAlgo
import ClassificationAlgo

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# CORS(app)

f = "dataset.csv"
df = pd.read_csv(f)
dummy = df.copy()
TARGET = ""
FEATURE = []

X_train, X_test, y_train,y_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

@app.route('/api/',methods=['GET'])
def home():
    return jsonify({"message":["working","person"]})

@app.route("/api/df/colsdata",methods=["POST","GET"])
def colsData():
    if request.method == "POST":
        data = request.get_json()
        cols = data.get('cols')
        data_val = dummy[cols].head(5).to_dict(orient='list')
        return jsonify({"data":data_val,"shape":dummy.shape})


@app.route("/api/df/datatypechange",methods=["POST","GET"])
def dataTypeChange():
    if request.method == 'GET':
        dtypes = {key: str(value) for key, value in dummy.dtypes.to_dict().items()}
        return jsonify({'dtypes':dtypes})

    if request.method == 'POST':
        data = request.get_json()
        key = data.get('key')
        dtype = data.get('dtype').lower()
        dtypes = {key: str(value) for key, value in dummy.dtypes.to_dict().items()}

        if 'int' in dtype:
            try:
                dummy[key] = dummy[key].astype(int)
                print("Key: ",key,"Integer",dummy[key])
            except:
                return {'changed':False,"msg":f"{key} can not be casted to {dtype}","dtypes":dtypes,"key":key,"dtype":dtype}
        elif 'float' in dtype:
            try:
                dummy[key] = dummy[key].astype(float)
            except:
                return {'changed':False,"msg":f"{key} can not be casted to {dtype}","dtypes":dtypes,"key":key,"dtype":dtype}

        elif 'object' in dtype:
            try:
                dummy[key] = dummy[key].astype(str)
            except:
                return {'changed':False,"msg":f"{key} can not be casted to {dtype}","dtypes":dtypes,"key":key,"dtype":dtype}
        else:
            print(f"Fail to casted {key} to {dtype} datatype")
            return {'changed':False,"msg":f"{key} can not be casted to {dtype}","dtypes":dtypes,"key":key,"dtype":dtype}
        
        print(f"{key} casted to {dtype} datatype")
        return {'successful':True,'dtypes':dtypes,"msg":"Successfully Data type changed","key":key,"dtype":dtype}

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
    global df
    data = dummy.dropna()
    cols = request.args.get('cols')
    if cols:
        cols = cols.split(',')
        data_val = data[cols].head(int(count)).to_dict(orient='list')
        return jsonify({"data":data_val,"shape":df.shape})
    
    cols = list(data.columns)
    data_val = data[cols].head(int(count)).to_dict(orient='list')
    return jsonify({"data":data_val,"shape":df.shape,"cols":cols})

@app.route('/api/df/dfuniquecount',methods=['GET'])
def dfuniquecount():
    return jsonify({"data":dummy.nunique(axis=0).to_dict(),"shape":dummy.shape})

@app.route('/api/dfcols',methods=['GET'])
def dfcols():
    print("COLUMNS")
    return jsonify({"cols":list(df.columns)})

@app.route('/api/encode-columns',methods=['GET','POST'])
def data_encoding():
    if request.method == 'GET':
        return {"columns":list(dummy.columns)}
    
@app.route('/api/df/encode-df',methods=['GET','POST'])
def encode_df():
    data = request.get_json()
    key = data.get('cols')
    ord_enc = OrdinalEncoder()
    dummy[key] = ord_enc.fit_transform(dummy[key])
    return jsonify({"cols":list(key),"new_data":dummy[key].head().to_dict(orient='list')})

@app.route("/api/df/missingdata",methods=['GET'])
def missingData():
    global dummy
    dtypes = {key: str(value) for key, value in dummy.dtypes.to_dict().items()}
    return {"missing":dummy.isna().sum().to_dict(),"dtypes":dtypes}

@app.route("/api/select-target-feature/<target>",methods=['POST','GET'])
def selecttarget(target):
    TARGET = target
    print("Target Feature is Set : ",target)
    return jsonify({"msg":"Target Feature is Set "+target,"update":True})

@app.route('/api/feature-target',methods=['POST'])
def featureTarget():
    global FEATURE,TARGET
    data = request.get_json()
    FEATURE = data.get('feature')
    TARGET = data.get('target')
    print("Feature : ",FEATURE)
    print("Target : ",TARGET)
    return jsonify({"msg":"done"})


@app.route("/api/train-test-split",methods=['POST','GET'])
def traintestsplit():
    global X_train,X_test,y_train,y_test
    data = request.get_json()
    randonState = int(data.get('randomstate'))
    shuffle = data.get('shuffle')
    trainSize = int(data.get('trainsize'))
    X_train, X_test, y_train, y_test = train_test_split(
        dummy[list(FEATURE)],
        dummy[[TARGET]],
        random_state=randonState,
        train_size=trainSize/100,
        shuffle=True
    )    
    return jsonify({'msg':'train test split done',"trainshape":X_train.shape,"testshape":X_test.shape})

@app.route("/api/regression-classification-algo",methods=["GET"])
def regressionAlgo():
    reg_algo = list(RegressionAlgo.regression_algorithms.keys())
    clf_algo = list(ClassificationAlgo.classification_algorithms.keys())
    return jsonify({'success':True,'regression':reg_algo,'classification':clf_algo})






if __name__ == "__main__":
    app.run(debug=True,port=5001)