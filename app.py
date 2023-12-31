from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 요청에서 JSON 데이터를 가져옴
        data = request.get_json()
        pool = data['pool']
        values = data['values']
        
        # 모델과 스케일러 로드
        model = load_model('model/{}_FRI.h5'.format(pool))
        sc = joblib.load('scaler/{}_FRI.pkl'.format(pool))

        # 데이터 처리 - 입력 데이터를 NumPy 배열로 변환하고 모델 입력 형식에 맞게 재구성
        X_test = np.array(values)
        X_test = X_test.reshape(-1, 1)
        X_test = sc.transform(X_test)
        X_test = X_test.reshape((X_test.shape[0] // 90), X_test.shape[0], X_test.shape[1])

        # 모델에 데이터를 입력하여 예측
        predicted = model.predict(X_test)

        # 예측 결과를 원래 스케일로 역변환
        predicted_inversed = sc.inverse_transform(predicted)

        # 입력 데이터와 예측 결과를 JSON 형식으로 리턴
        return jsonify({
            'input_values': values,
            'input_values_length': len(values),
            'pool': pool,
            'predicted': predicted_inversed.tolist()
        })
    
    except Exception as e:
        # 에러 발생 시 간단한 에러 메시지 반환
        return "error"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
