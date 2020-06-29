from server_rpc.pb import SimpleChatbot_pb2_grpc
from server_rpc.pb.SimpleChatbot_pb2 import ChatReply


class SimpleChatBotServicerImp(SimpleChatbot_pb2_grpc.SimpleChatBotServerServicer):
    def __init__(self, processor):
        self.processor = processor

    def Chat(self, request, context):
        print("Seq2SeqChatBot:New Req  ids %s" % request.ids)
        rep = ChatReply()
        rep.ids = request.ids
        rep.data = self.processor.process(request.data)
        return rep
