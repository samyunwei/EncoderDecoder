import grpc
import server_rpc.pb.SimpleChatbot_pb2_grpc
import server_rpc.pb.SimpleChatbot_pb2
from concurrent import futures
import logging
import server_rpc.SimpleChatBotServerImp
import server_rpc.SimpleChatProcessor
import sys

ADDR = "[::]:50052"


def serve():
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    processor = server_rpc.SimpleChatProcessor.SimpleChatProcessor()
    imp = server_rpc.SimpleChatBotServerImp.SimpleChatBotServicerImp(processor)
    server_rpc.pb.SimpleChatbot_pb2_grpc.add_SimpleChatBotServerServicer_to_server(imp, s)
    s.add_insecure_port(ADDR)
    s.start()
    print("SimpleChatBot Begin to Serve at:%s" % ADDR)
    s.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
