from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# 모델 로드
model = load_model('model/A_FRI.h5')

# 스케일러 로드
scaler = joblib.load('scaler/A_FRI.pkl')

@app.route('/sum', methods=['GET'])
def sum():
    a = request.args.get('a', default=0, type=float)
    b = request.args.get('b', default=0, type=float)
    return {'a': a, 'b': b, 'result': a + b}

@app.route('/test', methods=['POST'])
def test():
    # 요청에서 JSON 데이터를 가져옴
    data = request.get_json()

    # 데이터 처리 - 입력 데이터를 NumPy 배열로 변환하고 모델 입력 형식에 맞게 재구성
    X_test = np.array(data).reshape(1, 90, 1)  # 샘플 1개, 타임스탭 90, 특성 1개

    # 모델에 데이터를 입력하여 예측
    predicted = model.predict(X_test)

    # 예측 결과를 원래 스케일로 역변환
    predicted_inversed = scaler.inverse_transform(predicted)

    # 입력 데이터와 예측 결과를 JSON 형식으로 리턴
    return jsonify({
        'input_data': data,
        'predicted': predicted_inversed.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
