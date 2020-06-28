import distutils.util
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def argument_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group('run mode')
    # support
    group.add_argument('--name', type=str, required=True, default='senti2class')

    group.add_argument('--train', action='store_true', help='train the model')
    group.add_argument('--test', action='store_true', help='test the model on dev data')
    group.add_argument('--predict', action='store_true',
                       help='predict for test data with trained model')

    group.add_argument('--export', action='store_true', help='export model to inference')

    # Train config
    group = parser.add_argument_group('Train config')
    group.add_argument('--model_name', type=str, default='bow', help='train model type')
    group.add_argument('--batch_size', type=int, default=512,
                       help='The sequence number of a mini-batch data.')
    group.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate used to train the model.")
    group.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rprop'], help='optimizer type')
    group.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    group.add_argument('--drop_rate', type=float, default=0, help='dropout rate')
    group.add_argument("--use_gpu", type=distutils.util.strtobool, default=True,
                       help="Whether to use gpu.")

    group.add_argument('--max_epoch', type=int, default=5, help='train epochs')
    group.add_argument('--save_threold', type=float, default=0.68, help='train epochs')
    group.add_argument('--random_seed', type=int, default=42)
    group.add_argument("--test_step", type=int, default=100,
                       help="log the train loss every n batches.")

    group.add_argument("--min_freq", type=int, default=10,
                       help="the most word times")

    # model save dir
    group = parser.add_argument_group('dir config')
    group.add_argument("--load_dir", type=str, default=None, help="load model")
    group.add_argument("--save_dir", type=str, default="./stat_dict",
                       help="Specify the path to save trained models.")

    # token config
    group = parser.add_argument_group('token config')
    group.add_argument('--maxlen', type=int, default=25, help='max sentence length')
    group.add_argument('--vocab_path', type=str, default='data/vocab/vocab.txt', help='vocab path')
    group.add_argument('--pretrain', type=str, default=None,
                       help='pretrain embedding path')
    group.add_argument('--glove', type=str, default=None,
                       help='glove embedding path')

    # dataset
    group = parser.add_argument_group('dataset config')
    group.add_argument('--data_dir', type=str, default='/home/demo1/womin/datasets/')

    group.add_argument('--seed', type=int, default=123, help='random seed')
    group.add_argument('--log_dir', type=str, default='./logs', help='log dir')
    group.add_argument('--export_path', type=str, default=None, help='export model to path')

    # Network config
    group = parser.add_argument_group('Network config')
    group.add_argument('--tgt_size', type=int, default=6, help='target size')
    group.add_argument('--embed_size', type=int, default=300, help='embed size')
    group.add_argument('--hidden_size', type=int, default=200, help='hidden size')
    group.add_argument('--num_layers', type=int, default=1, help='rnn num_layers')
    group.add_argument('--bidirectional', type=str2bool, default=False, help='rnn num_layers')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    print(args)
