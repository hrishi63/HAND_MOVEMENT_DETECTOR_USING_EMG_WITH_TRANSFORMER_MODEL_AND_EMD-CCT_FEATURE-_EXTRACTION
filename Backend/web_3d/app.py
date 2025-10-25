from flask import Flask, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # allows React on localhost:3000

LOCAL_PREDICT = "http://127.0.0.1:5000/prediction"

@app.route("/api/clench")
def api_clench():
    try:
        r = requests.get(LOCAL_PREDICT, timeout=0.2)
        return jsonify({"clench": bool(r.json()[0])})
    except Exception:
        return jsonify({"clench": False})

if __name__ == "__main__":
    app.run(port=5001, debug=False)