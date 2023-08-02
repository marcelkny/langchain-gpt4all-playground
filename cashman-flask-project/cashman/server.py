from flask import Flask, jsonify, request, json
from ai import ai_chat
from config.apikey import APIKEY 

app = Flask(__name__)

def isAuthorized(headers):
    auth = headers.get("X-APIKEY")
    if auth == APIKEY:
        return True
    else:
        return False


@app.route('/')
async def get_incomes():
    if isAuthorized(request.headers) == True:
        aiResponse = await ai_chat()
        response = app.response_class(
            response=json.dumps(aiResponse),
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    


    

@app.route('/ai', methods=['POST'])
async def post_income():
    if isAuthorized(request.headers) == True:
        aiResponse = await ai_chat()
        response = app.response_class(
            response=json.dumps(aiResponse),
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401