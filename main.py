from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# CORS(app)

f = "dataset.csv"
df = pd.read_csv(f)


@app.route('/api/',methods=['GET'])
def home():
    return jsonify({"message":["working","person"]})

@app.route('/api/df/<count>',methods=['GET'])
def dataframe(count:int = 5):
    print("COUNT : ",count)
    data = df.dropna()
    data_val = data.head(int(count)).to_dict(orient='list')
    return {"data":data_val,"shape":data.shape}

@app.route("/api/df/missingdata",methods=['GET'])
def missingData():
    dtypes_series = df.dtypes
    serializable_dtypes = {col: str(dtype) for col, dtype in dtypes_series.items() if not pd.api.types.is_extension_type(dtype)}
    return {"missing":df.isna().sum().to_dict(),"dtypes":serializable_dtypes}


if __name__ == "__main__":
    app.run(debug=True,port=5001)