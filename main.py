from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
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
hist = ''
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
    try:
        data = request.get_json()
        randonState = int(data.get('randomstate'))
        shuffle = data.get('shuffle')
        trainSize = int(data.get('trainsize'))
    except Exception as e:
        print("Error : ",e)
        return jsonify({'success':False,'message':"Error : input value is not filled","trainshape":X_train.shape,"testshape":X_test.shape})

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            dummy[list(FEATURE)],
            dummy[[TARGET]],
            random_state=randonState,
            train_size=trainSize/100,
            shuffle=True
        )    
    except Exception as e:
        print("Error : ",e)
        return jsonify({'success':False,'message':"Error : Feature and Target is not selected","trainshape":X_train.shape,"testshape":X_test.shape})
    
    print("train test split done")
    return jsonify({'success':True,'message':'train test split done',"trainshape":X_train.shape,"testshape":X_test.shape})


@app.route("/api/regression-classification-algo",methods=["GET"])
def regressionAlgo():
    reg_algo = list(RegressionAlgo.regression_algorithms.keys())
    clf_algo = list(ClassificationAlgo.classification_algorithms.keys())
    return jsonify({'success':True,'regression':reg_algo,'classification':clf_algo})

@app.route("/api/model-train-algo",methods=["GET","POST"])
def trainAlgo():
    global hist
    data = request.get_json()
    algoType = data.get('algoType')
    algo = data.get('algo')
    print("Algo : ",algo)
    print("AlgoType : ",algoType)

    # try:
    if algoType == 'regression' and algo in list(RegressionAlgo.regression_algorithms.keys()):
        model = RegressionAlgo.regression_algorithms[algo]
        hist = model.fit(X_train,y_train)
    elif algoType == 'classification' and algo in list(ClassificationAlgo.classification_algorithms.keys()):
        model = ClassificationAlgo.classification_algorithms[algo]
        hist = model.fit(np.array(X_train),np.ravel(y_train))
    else:
        print("Not Found : ",algo," Type : ",algoType)
        return jsonify({'success':False,"message":"Not Algorithm is selected",'features':FEATURE,'target':TARGET})
    # except:
    #     print("Data set not splitted")
    #     return jsonify({'success':False,"message":"train test split is not performed",'features':FEATURE,'target':TARGET})

    return jsonify({'success':True,"message":"successful",'features':FEATURE,'target':TARGET})



@app.route("/api/mode-predict",methods=["GET","POST"])
def modelPredict():
    data = request.get_json()
    f = data.get('featureValue')
    print("Feature Value : ",f)
    int_feature = [ int(i) for i in f ]
    print("feature : ",int_feature)
    pred = hist.predict([int_feature])
    print("Prediction : ",str(pred))
    return jsonify({'predict':round(np.array(pred).ravel()[0])})





if __name__ == "__main__":
    app.run(debug=True,port=5001)