from flask import Flask, jsonify
from main import run_model, print_dict
app = Flask(__name__)

#Routes
# to run in terminal, type : curl http://127.0.0.1:3001/ | jq
@app.route('/', methods=["GET"])
def run():
    '''Run the DL model'''
    result = run_model()
    print_dict(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)