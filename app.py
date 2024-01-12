from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 요청에서 JSON 데이터를 가져옴
        data = request.get_json()
        pool = data['pool']
        values = data['values']

        if (pool == 'A' or pool == 'G' or pool == 'J' or pool == 'L' or pool == 'M' or pool == 'N' or pool == 'P' or pool == 'Q' or pool == 'S' or pool == 'T' or pool == 'U' or pool == 'V'):
            # 각 키에 해당하는 값을 추출
            data1 = [value['data1'] for value in values]
            data2 = [value['data2'] for value in values]
            data3 = [value['data3'] for value in values]
            data4 = [value['data4'] for value in values]
            
            # 데이터셋 만들기
            data_set = np.column_stack((data1, data2, data3, data4))
        elif (pool == 'B'):
            return

        # 날짜 키에 해당하는 값을 추출
        dateTime = [value['dateTime'] for value in values]

        # 시간값 변환
        dateTime_trans = []
        for i in range(len(dateTime)):
            year = dateTime[i][0]
            month = f'0{dateTime[i][1]}' if dateTime[i][1] < 10 else dateTime[i][1]
            day = f'0{dateTime[i][2]}' if dateTime[i][2] < 10 else dateTime[i][2]
            hour = f'0{dateTime[i][3]}' if dateTime[i][3] < 10 else dateTime[i][3]
            minute = f'0{dateTime[i][4]}' if dateTime[i][4] < 10 else dateTime[i][4]
            
            dateTime_trans.append(f'{year}-{month}-{day}T{hour}:{minute}')

        # 모델과 스케일러 로드
        model = load_model('lstm/model/{}.keras'.format(pool))
        with open('lstm/scaler/{}_mms.pkl'.format(pool), 'rb') as file:
            mms = pickle.load(file)

        with open('lstm/scaler/{}_sts.pkl'.format(pool), 'rb') as file:
            sts = pickle.load(file)

        # 데이터 스플릿
        time_steps = 6 * 24
        X_test = []
        y_test = []
        for i in range(time_steps, data_set.shape[0]):
            X_test.append(data_set[i-time_steps:i, :])
            y_test.append(data_set[i, -1:])
        X_test, y_test = np.array(X_test), np.array(y_test)

        # 스케일링
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_test_scaled = mms.transform(X_test)
        y_test_scaled = sts.transform(y_test)

        # 모델에 데이터를 입력하여 예측
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], time_steps, data_set.shape[1])
        predicted_data = model.predict(X_test_scaled)

        # 예측 결과를 원래 스케일로 역변환
        predicted_data_inverse = sts.inverse_transform(predicted_data)

        # MSE 계산
        mse = mean_squared_error(y_test_scaled, predicted_data)

        # MAE 계산
        mae = mean_absolute_error(y_test_scaled, predicted_data)

        # R^2 계산
        r2 = r2_score(y_test_scaled, predicted_data)

        # 입력 데이터와 예측 결과를 JSON 형식으로 리턴
        return jsonify({
            'actual': y_test.tolist(),
            'dateTime': dateTime_trans[-len(predicted_data_inverse):],
            'mae': mae,
            'mse': mse,
            'pool': pool,
            'predict': predicted_data_inverse.tolist(),
            'r2': r2,
        })
    
    except Exception as e:
        # 에러 발생 시 간단한 에러 메시지 반환
        return "error"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
