# -*- coding:utf-8 -*-
# Date: 2019/7/18 14:26
# Author: xuxiaoping
# Desc: Train Program

import datetime
import logging
import os
import time

import numpy as np
import torch
import torch as t
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optimizers
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score, precision_score
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchtext.data import Iterator, BucketIterator
from torchtext import data
from args import argument_parser
from data.dataset import SnDataset,MyDataset
from models import SN_MODELS
from data.tokenizer import load_vocab, load_pretrain_embedding
import torch.nn.functional as F
import spacy
spacy_en = spacy.load('en')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from data.text_utils import tokenizer


def default_load_data(mode='train', vocab=None, args=None):
    if mode == 'train':
        dataset = SnDataset(data_path=args.dataset_files['train'],
                            vocab=vocab, opt=args)
        #sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)





    elif mode == 'dev':
        dataset = SnDataset(data_path=args.dataset_files['dev'],
                            vocab=vocab, opt=args)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    elif mode == 'test':
        dataset = SnDataset(data_path=args.dataset_files['test'],
                            vocab=vocab, opt=args)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    else:
        raise ValueError('invalid mode!')

    return dataloader

def write_config(args, logger):
    configs = vars(args)
    config_path = 'config/train_{}_{}.conf'.format(args.name, args.model_name)
    with open(config_path, 'w', encoding='utf-8') as fout:
        for config, value in configs.items():
            fout.write('{}={}\n'.format(config, value))

    logger.info('save config to {}'.format(config_path))


def check_gpu(args, logger):
    use_gpu = False
    if args.use_gpu and torch.cuda.is_available():
        cudnn.benchmark = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        use_gpu = True
    else:
        logger.info('GPU is highly recommend!')

    args.device = torch.device('cuda' if use_gpu else 'cpu')


def get_logger(args):
    logger = logging.getLogger("classify")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_path = '{}/{}_{}_{}.txt'.format(args.log_dir, args.name, args.model_name, log_time)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class TorchProgram:

    def __init__(self, args, logger):
        model_class = SN_MODELS[args.model_name]

        #logger.info('load vocab for {}'.format(args.vocab_path))
        #vocab = load_vocab(args.vocab_path)
        #self.vocab = vocab



        TEXT = data.Field(sequential=True, tokenize=tokenizer, batch_first=True,
                          lower=True, fix_length=args.maxlen, init_token="<sos>", eos_token="<eos>")
        LENGTH = data.Field(sequential=False, use_vocab=False)

        train = MyDataset(args.dataset_files['train'], text_field=TEXT, len_field=LENGTH, test=False, aug=0)

        TEXT.build_vocab(train, min_freq=args.min_freq, max_size=10000 )
        LENGTH.build_vocab(train)




        # 统计词频
        #Todo ？
       # TEXT.vocab.freqs.most_common(args.most_common)

        print(TEXT.vocab.freqs.most_common(20))

        if args.pretrain:
            logger.info('Load pretrain embeddings from {}'.format(args.pretrain))
            embeddings = torch.tensor(
                np.load(args.pretrain, allow_pickle=True)["embeddings"].item().weight)
        else:
            embeddings = np.random.random((len(TEXT.vocab.itos), args.embed_size))
        args.TEXT = TEXT

        encoder = SN_MODELS["encoder"](embeddings, args)
       # atten = SN_MODELS["attention"](args.hidden_size * 4, 300)
        #Todo need to judge if use atten
        #atten = SN_MODELS["attention"](args.hidden_size, 2, "general")
        atten = SN_MODELS["attention"](args.hidden_size, "general")

        decoder = SN_MODELS["decoder"](embeddings, args, atten)

        self.model = model_class(encoder, decoder, args)

        self.inputs_cols = ['text_raw_indices', 'lengths'] #?好像可去掉




        train_iter= BucketIterator(
            train,  # 构建数据集所需的数据集
            #Todo batch
            batch_size=args.batch_size,
            device=-1,  # 如果使用gpu，此处将-1更换为GPU的编号
            sort_key=lambda x: x.source_length,
            # the BucketIterator needs to be told what function it should use to group the data.
            sort_within_batch=True
        )


        # build loader
        if args.train:
            #确定logdir
            self.log_writer = SummaryWriter(logdir= "scalar",comment=args.model_name)

            #self.trainloader = load_data_func('train', vocab, args)
            #self.devloader = load_data_func('dev', vocab, args)


            self.trainloader = train_iter


        if args.test:
            #self.testloader = load_data_func('dev', vocab, args)

            self.testloader = None


        self.model.to(args.device)

        logger.info(self.model)
        self.start_epoch, self.end_epoch = 0, args.max_epoch
        if args.load_dir and os.path.exists(args.load_dir):
            checkpoint = torch.load(args.load_dir)
            self.start_epoch = checkpoint['epoch']
            self.end_epoch = self.start_epoch + args.max_epoch
            #self.model.load_state_dict(checkpoint['model'])
            self.model.load_state_dict(checkpoint['model'])

        self.args = args
        self.logger = logger


    def train(self):
        optimizer = optimizers.Adam(params=self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)

        criterion = SN_MODELS["criterion"]().to(self.args.device)

        max_test_pre = 0
        best_mode_path = None

        global_step = 1
        global_dev_step = 1

        min_loss = float('inf')
        for epoch in range(self.start_epoch, self.end_epoch):
            total_correct, total_num = 0, 0
            total_loss = 0


            for batch_id, batch_data in enumerate(self.trainloader, 1):
                self.model.train()
                start_time = time.time()

                optimizer.zero_grad()



                sources = batch_data.source_text.to(self.args.device)
                sources_length = batch_data.source_length.to(self.args.device)
                targets = batch_data.target_text.to(self.args.device)
                target_length = batch_data.target_len.to(self.args.device)


                output, attention = self.model(sources, sources_length, targets, target_length)
                #选择没有pad的数据
                batch = output.shape[0]
                target_max_len = t.max(target_length).item()
                mask = t.arange(target_max_len).repeat(batch, 1).to(self.args.device) < target_length.view(-1, 1).expand(-1,
                                                                                                            target_max_len)
                mask = mask.float()
                loss = criterion(output, targets, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()


                total_loss += loss.item()
                self.log_writer.add_scalar('scalar/loss', loss.item(), global_step)
                end_time = time.time()



                global_step += 1

            self.logger.info(
                "Epoch:{}, Batch:{}, Time:{:.2f}, Total_Loss:{:.4f},"
                .format(epoch, batch_id, end_time - start_time, total_loss))

            #self.test(self.devloader)

            if total_loss < min_loss:
                min_loss = total_loss

                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                best_mode_path = '{}/{}_{}_{:.4f}.pkl'.format(self.args.save_dir,
                                                          self.args.model_name,
                                                          str(epoch), total_loss)

                text_path = '{}/{}_{}_text.pkl'.format(self.args.save_dir,
                                                              self.args.model_name,
                                                              str(epoch), total_loss)

                torch.save({
                    'model': self.model.state_dict(),
                    'epoch': epoch
                }, best_mode_path)

        import dill
        with open("seq2seq/TEXT.Field", "wb")as f:
            dill.dump(self.args.TEXT, f)

        self.logger.info('save model to {}'.format(best_mode_path))

        self.log_writer.close()

    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            criterion = SN_MODELS["criterion"]().to(self.args.device)
            loss = None
            for batch_id, batch_data in enumerate(dataloader, 1):
                sources = batch_data['source_raw_indices'].to(self.args.device)
                sources_length = batch_data['source_length'].to(self.args.device)
                targets = batch_data['target_raw_indices'].to(self.args.device)
                target_length = batch_data['target_length'].to(self.args.device)


                pred, attention = self.model(sources, sources_length, targets, target_length)
                #选择没有pad的数据
                mb_out_mask = (t.arange(target_length.max().item())[None, :]).to(self.args.device)  < target_length[:, None]

                mb_out_mask = mb_out_mask.float()
                if type(attention) != None:
                    pred_trunked = pred[:, :target_length.max().item()]
                    pred = pred_trunked
                targets = targets[ :, : target_length.max().item()  ]
                loss = criterion(pred, targets, mb_out_mask)

            self.logger.info( "Test Loss:{:.4f},"
                        .format(loss.item()))

        self.model.train()


    def test_seq2seq(self, vocab):
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(self.testloader, 1):
                sources = batch_data['source_raw_indices'].to(self.args.device)
                sources_length = batch_data['source_length'].to(self.args.device)
                targets = batch_data['target_raw_indices'].to(self.args.device)
                target_length = batch_data['target_length'].to(self.args.device)

                new_targets = t.full((sources_length.shape[0],1), vocab.get_id("START")).long().to(self.args.device)
                new_target_length = t.full((sources_length.shape[0],1), 1).long().to(self.args.device)

                #pred, attention = self.model(sources, sources_length, targets, target_length)
                self.model.translate(sources, sources_length, targets, target_length,target_length.max().item(), new_targets, new_target_length, vocab)


    def export(self, export_path):
        torch.save(self.model, export_path)


def main():
    args = argument_parser()

    dataset_files = {
        'seq2seq': {
            'train': args.data_dir + '/formatted_movie_lines.txt',
            'dev': args.data_dir + '/formatted_movie_lines.txt',
            'bak':  '/formatted_movie_lines_bak.txt'
        },

    }

    save_dirs = {
        'senti2class': 'senti2class_models',
        'senti3class': 'senti3class_models',
        'senti6class': 'senti6class_models'
    }

    args.dataset_files = dataset_files[args.name]
    #args.tgt_size = tgt_sizes[args.name]
    #args.save_dir = save_dirs[args.name]

    logger = get_logger(args)

    logger.info(args)
    check_gpu(args, logger)

    program = TorchProgram(args, logger)

    if args.train:
        program.train()
        write_config(args, logger)

    if args.test and args.load_dir:
        program.test_seq2seq(program.vocab)




    if args.export:
        program.export(args.export_path)


if __name__ == "__main__":
    main()
