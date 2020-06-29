#!/usr/bin/env bash
python -m grpc_tools.protoc -I../server_rpc/pb --python_out=../server_rpc/pb --grpc_python_out=../server_rpc/pb ../server_rpc/pb/SimpleChatbot.proto
