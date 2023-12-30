from flask import Flask, request

app = Flask(__name__)

@app.route('/sum', methods=['GET'])
def sum():
    a = request.args.get('a', default=0, type=float)
    b = request.args.get('b', default=0, type=float)
    return {'a': a , 'b': b, 'result': a + b}

if __name__ == '__main__':
    app.run(debug=True)
