import os
from server.msg_server import return_msg
from server.model_server import model_msg
import numpy as np


class SimpleChatProcessor(object):
    def __init__(self):
        pass

    def process(self, msg):
        proba = np.random.normal(loc=0.0, scale=1.0, size=(1))[0]
        if proba > 0.6:
            intent = model_msg(msg)
        else:
            intent = return_msg(msg)
        return intent
