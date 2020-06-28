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

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)


@app.route('/receive', methods = ["GET"])
def receive_msg():
    msg = request.args.get("msg")
    intent = return_msg(msg)
    return jsonify({"intent":intent})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8002, debug=True)