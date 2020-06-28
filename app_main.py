# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 14:43
# @Author  : piguanghua
# @FileName: flask_mian.py
# @Software: PyCharm

from flask import Flask
from flask_socketio import SocketIO, emit
from flask import request
from flask import jsonify
from server.msg_server import return_msg
from server.model_server import model_msg

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)
import numpy as np

@app.route('/receive', methods = ["GET"])
def receive_msg():
    proba = np.random.normal(loc=0.0, scale=1.0, size=(1))[0]
    msg = request.args.get("msg")
    if proba > 0.6:
        intent = model_msg(msg)
    else:
        intent = return_msg(msg)
    return jsonify({"intent":intent})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=6006, debug=True)