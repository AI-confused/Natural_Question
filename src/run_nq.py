from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import time
import collections
import torch.nn as nn
from collections import defaultdict
import gc
import itertools
from multiprocessing import Pool
import functools
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from typing import Callable, Dict, List, Generator, Tuple
from torch.utils.data.distributed import DistributedSampler
# from tqdm.notebook import tqdm_notebook as tqdm
from tqdm import tqdm
from sklearn.metrics import f1_score
import json
import math
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig
from modeling_nq import BertForQuestionAnswering, loss_fn
from prepare_data_version2 import Example, Test_Example, convert_data, convert_test_data, Result, JsonChunkReader, Test_Json_Reaser
from itertools import cycle
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
MODEL_CLASSES = {
    'bert': (BertForQuestionAnswering, BertTokenizer),
}
logger = logging.getLogger(__name__)

valid_size = 0
train_size = 245899 - valid_size
test_size = 346

class TextDataset(Dataset):
    """Dataset for [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering).
    
    Parameters
    ----------
    examples : list of List[Example]
        The whole Dataset.
    """
    def __init__(self, examples: List[List[Example]]):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
      
    def __getitem__(self, index):
#         print('index:', index)
        annotated = list(
            filter(lambda example: example.class_label != 'unknown', self.examples[index]))
#         print('len(annotated)',len(annotated))
        if len(annotated) == 0:
            return random.choice(self.examples[index])
        return random.choice(annotated)
    
class TestDataset(Dataset):
    """
    Dataset for test data
    """
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
      
    def __getitem__(self, index):
        return self.examples[index]
    
    
def collate_fn(examples: List[Example]) -> List[List[torch.Tensor]]:
    # input tokens
#     print(len(examples))
    max_len = max([len(example.input_ids) for example in examples]) #batch
    tokens = np.zeros((len(examples), max_len), dtype=np.int64)
#     print(tokens)
    token_type_ids = np.ones((len(examples), max_len), dtype=np.int64)
    for i, example in enumerate(examples):
        row = example.input_ids
        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]
        token_type_ids[i, :len(row)] = token_type_id
    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens), #input_id
              torch.from_numpy(attention_mask), #mask_id
              torch.from_numpy(token_type_ids)] #segment_id

    # output labels
    all_labels = ['long', 'no', 'short', 'unknown', 'yes']
    start_positions = np.array([example.start_position for example in examples])
    end_positions = np.array([example.end_position for example in examples])
    class_labels = [all_labels.index(example.class_label) for example in examples]
    start_positions = np.where(start_positions >= max_len, -1, start_positions)
    end_positions = np.where(end_positions >= max_len, -1, end_positions)
    labels = [torch.LongTensor(start_positions),
              torch.LongTensor(end_positions),
              torch.LongTensor(class_labels)]

    return [inputs, labels]
    

def eval_collate_fn(examples: List[Example]) -> Tuple[List[torch.Tensor], List[Example]]:
    # input tokens
    max_len = max([len(example.input_ids) for example in examples])
    tokens = np.zeros((len(examples), max_len), dtype=np.int64)
    token_type_ids = np.ones((len(examples), max_len), dtype=np.int64)
    for i, example in enumerate(examples):
        row = example.input_ids
        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]
        token_type_ids[i, :len(row)] = token_type_id
    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask),
              torch.from_numpy(token_type_ids)]

    return inputs, examples


def test_collate_fn(examples: List[Test_Example]) -> Tuple[List[torch.Tensor], List[Test_Example]]:
    # input tokens
    max_len = max([len(example.input_ids) for example in examples])
    tokens = np.zeros((len(examples), max_len), dtype=np.int64)
    token_type_ids = np.ones((len(examples), max_len), dtype=np.int64)
    for i, example in enumerate(examples):
        row = example.input_ids
        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]
        token_type_ids[i, :len(row)] = token_type_id
    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask),
              torch.from_numpy(token_type_ids)]

    return inputs, examples


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return f1_score(labels,outputs,labels=[0,1,2],average='macro')
        
        
class Test_Result(object):
    def __init__(self):
        self.examples = {}
        self.results = {}
        self.logits = {}
        self.class_pred = {}
        self.class_labels = ['LONG', 'NO', 'SHORT', 'UNKNOWN', 'YES']
        
    @staticmethod
    def is_valid_index(example: Test_Example, index: List[int]) -> bool:
        """Return whether valid index or not.
        """
        start_index, end_index = index
        if start_index > end_index:
            return False
        if start_index <= example.question_len + 2:
            return False
        return True
    
    def update(self, examples, logits, indices, class_pred):
        class_pred = torch.max(class_pred, dim=1)[1].numpy() # (batch,)
        for i, example in enumerate(examples):#batch
            if not example.example_id in self.examples.keys():
                self.examples[example.example_id] = []
            self.examples[example.example_id].append(example)
            if not example.example_id in self.results.keys():
                self.results[example.example_id] = []
            self.results[example.example_id].append(indices[i])
            if not example.example_id in self.logits.keys():
                self.logits[example.example_id] = []
            self.logits[example.example_id].append(logits[i])
            if not example.example_id in self.class_pred.keys():
                self.class_pred[example.example_id] = []
            self.class_pred[example.example_id].append(self.class_labels[class_pred[i]])
#         print(self.examples, self.results, self.logits, self.class_pred)
    
    def generate_prediction(self):
#         answers = []
        long_answers = {}
        short_answers = {}
        class_answers = {}
        for i, item in enumerate(self.results.keys()):
#             print(self.logits[item])
#             print(item) # example_id
            sorted_index = sorted(range(len(self.logits[item])), key=lambda k: self.logits[item][k], reverse=True)
#             print(sorted_index)
#             answer = {}
            for j in sorted_index:
                if self.class_pred[item][j] in ['YES','NO']:
                    short_answer = self.class_pred[item][j]
                    long_answer_s = self.examples[item][j].candidate[2]
                    long_answer_e = self.examples[item][j].candidate[3]
                    long_answer = str(long_answer_s)+':'+ str(long_answer_e)
                    class_answer = self.class_pred[item][j]
                    break
                elif self.class_pred[item][j]=='SHORT' and self.is_valid_index(self.examples[item][j], self.results[item][j]):
                    short_answer_s = self.examples[item][j].tokenized_to_original_index[self.results[item][j][0]-self.examples[item][j].question_len-2+self.examples[item][j].doc_token_start]
                    short_answer_e = self.examples[item][j].tokenized_to_original_index[self.results[item][j][1]-self.examples[item][j].question_len-2+self.examples[item][j].doc_token_start]
                    long_answer_s = self.examples[item][j].candidate[2]
                    long_answer_e = self.examples[item][j].candidate[3]
                    long_answer = str(long_answer_s)+':'+ str(long_answer_e)
                    short_answer = str(short_answer_s)+':'+ str(short_answer_e)
                    class_answer = self.class_pred[item][j]
                    break
                elif self.class_pred[item][j]=='LONG' and self.is_valid_index(self.examples[item][j], self.results[item][j]):
                    short_answer = None
                    long_answer_s = self.examples[item][j].candidate[2]
                    long_answer_e = self.examples[item][j].candidate[3]
                    long_answer = str(long_answer_s)+':'+str(long_answer_e)
                    class_answer = self.class_pred[item][j]
                    break
                elif self.class_pred[item][j]=='UNKNOWN' and not self.is_valid_index(self.examples[item][j], self.results[item][j]):
                    short_answer = None
                    long_answer = None
                    class_answer = self.class_pred[item][j]
                    break
#             print(short_answer, long_answer)
            long_answers[item] = long_answer
            short_answers[item] = short_answer
            class_answers[item] = class_answer
            
        return (long_answers, short_answers, class_answers)

def write_csv_line(content, csv_file, flag_row=1, flag_title=0):
    if flag_title:
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['example_id', 'PredictionString'])
    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        if flag_row:
            writer.writerows(content)
        else:
            writer.writerow(content)


def write_answer_public(long_answers, short_answers, sample_file, submit_file):
    sample_submission = pd.read_csv(sample_file)
    long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)
    short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)
    sample_submission.loc[sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings
    sample_submission.loc[sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings
    sample_submission.to_csv(submit_file, index=False)
    print('write done')

        
def write_answer_private(long_answers, short_answers, submit_file):
    contents = []
    for _ in long_answers.keys():
        content_l = [_+'_long', long_answers[_]]
        content_s = [_+'_short', short_answers[_]]
        contents.append(content_l)
        contents.append(content_s)
    write_csv_line(contents, submit_file, flag_title=1)
        
        
        

def main():
    #argparse start
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    ## Other parameters
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer, than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_question_len", default=64, type=int,
                        help="")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="")
    parser.add_argument("--chunksize", default=1000, type=int,
                        help="")
    parser.add_argument("--num_labels", default=5, type=int,
                        help="")
    parser.add_argument("--epoch", default=1, type=int,
                        help="")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
#     parser.add_argument("--train_steps", default=-1, type=int,
#                         help="")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    #argparse end
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # Set seed
    set_seed(args)
    
    try:
        os.makedirs(args.output_dir)
    except:
        pass
    
    #define nq model
    tokenizer = BertTokenizer.from_pretrained(os.path.join('./bert_large/', 'vocab.txt'), do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(os.path.join('./bert_large/', 'bert_config.json'), num_labels=args.num_labels)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
   

    if args.do_train:
        model = BertForQuestionAnswering.from_pretrained(args.model_name_or_path, config=config)
        model = model.to(args.device)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        #CUDA distribute
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Prepare data loader
        train_dir = os.path.join(args.data_dir, 'train.jsonl')
        dev_dir = os.path.join(args.data_dir, 'dev_example.jsonl')
        dev_size = len(open(dev_dir,'r').readlines()) 
        convert_func = functools.partial(convert_data,
                                 tokenizer=tokenizer,
                                 max_seq_len=args.max_seq_length,
                                 max_question_len=args.max_question_len,
                                 doc_stride=args.doc_stride)
        data_reader = JsonChunkReader(train_dir, convert_func, chunksize=args.chunksize)
        
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#                 scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.train_steps)
        
        
    
        global_step = 0
        tr_loss = 0
        nb_tr_steps = 0
        best_acc=0
        # new output file
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f:
            f.write('*'*80+'\n')
        bar_epoch = tqdm(range(args.epoch), total=args.epoch)
        for epoch in bar_epoch:
            bar_epoch.set_description('epoch_{}'.format(epoch))
            logger.info("***** training epoch: epoch_"+str(epoch)+" *****")
            bar_dataset = tqdm(data_reader, total=int(np.ceil(train_size / args.chunksize)))
            chunk_index = 1
            for examples in bar_dataset:
                bar_dataset.set_description('chunk_{}'.format(chunk_index))
                
                train_dataset = TextDataset(examples)
                logger.info('***** chunk_train_examples:' + str(len(train_dataset))+' *****')

                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_dataset)
                else:
                    train_sampler = DistributedSampler(train_dataset)
                train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps, collate_fn=collate_fn)
                num_train_optimization_steps = math.ceil(len(train_dataset)/args.train_batch_size)

                logger.info("***** Running training {}*****".format(chunk_index))
                logger.info("  Num examples = %d", len(train_dataset))
                logger.info("  Batch size = %d", args.train_batch_size)
                logger.info("  Num steps = %d", num_train_optimization_steps)
                chunk_index += 1
                model.train()
                bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
                train_dataloader = itertools.cycle(train_dataloader)
                for step in bar:
                    batch = next(train_dataloader)
#                     batch = tuple(t.to(device) for t in batch)
                    inputs, labels = batch
                    input_ids, mask_ids, segment_ids = inputs
                    y_label = (y.to(device) for y in labels)
                    y_pred = model(input_ids=input_ids.to(device), # tuple
                                   attention_mask=mask_ids.to(device),
                                   token_type_ids=segment_ids.to(device))
                    loss = loss_fn(y_pred, y_label)
                    if args.n_gpu > 1:
                        loss = loss.mean() # average on multi-gpu
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()
                    train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                    bar.set_description("loss {}".format(train_loss))
                    nb_tr_steps += 1
                    global_step += 1
                    loss.backward()
                    
                    if global_step % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    if global_step %(args.eval_steps*args.gradient_accumulation_steps)==0:
                        tr_loss = 0
                        nb_tr_steps = 0
                        logger.info("***** Report result *****")
                        logger.info("  %s = %s", 'global_step', str(global_step))
                        logger.info("  %s = %s", 'train loss', str(train_loss))
                
                    if args.do_eval and global_step %(args.eval_steps*args.gradient_accumulation_steps)==0:
                        dev_data_reader = JsonChunkReader(dev_dir, convert_func, chunksize=args.chunksize)
                        dev_bar = tqdm(dev_data_reader, total=math.ceil(dev_size/args.chunksize))
                        dev_chunk_index = 1
                        chunk_result = {'train_loss': train_loss,
                                  'global_step': global_step,
                                  'micro_F1': 0}
                        for dev_examples in dev_bar:
                            dev_bar.set_description('chunk_{}'.format(dev_chunk_index))
                            all_eval_examples = []
                            for _ in dev_examples:
                                for exam in _:
                                    all_eval_examples.append(exam)
                            dev_dataset = TestDataset(all_eval_examples)

                            logger.info('***** chunk_dev_examples: %d', len(dev_examples))
                            logger.info("***** Running evaluation {}*****".format(dev_chunk_index))
                            logger.info("  Num examples = %d", len(all_eval_examples))
                            logger.info("  Eval Batch size = %d", args.eval_batch_size)
                            dev_chunk_index += 1
                            # Run prediction for full data
                            dev_sampler = SequentialSampler(dev_dataset)
                            dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size, collate_fn=eval_collate_fn)   

                            model.eval()
                            with torch.no_grad():
                                result = Result(max_answer_length=args.max_answer_length)
                                for inputs, examples in tqdm(dev_dataloader):#batch
#                                     print(len(examples))
                                    input_ids, input_mask, segment_ids = inputs
                                    y_preds = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device), token_type_ids=segment_ids.to(device))
                                    start_preds, end_preds, class_preds = (p.detach().cpu() for p in y_preds) # (batch, seq) (batch, seq) (batch, 5)
                                    result.update(examples, start_preds.numpy(), end_preds.numpy(), class_preds.numpy())
                                    
                            microf1_score = result.score() # chunk_score
                            logger.info("  %s = %f", 'micro_f1', microf1_score)
                            #write to output file
                            with open(output_eval_file, "a") as writer:
                                for key in sorted(chunk_result.keys()):
                                    if key == 'micro_F1':
                                        writer.write("%s = %s\n" % (key, microf1_score))
                                    else:
                                        logger.info("  %s = %s", key, str(chunk_result[key]))
                                        writer.write("%s = %s\n" % (key, str(chunk_result[key])))
                                writer.write('*'*80)
                                writer.write('\n')
                        if microf1_score > best_acc:
                            best_acc = microf1_score
                            logger.info("  %s = %f", 'best_f1', best_acc)
                            print("Saving Model......")
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                                                                    
        del train_dataloader, train_dataset, dev_dataset, dev_dataloader, data_reader, dev_data_reader
        gc.collect()
    
#     if args.do_test:
#         args.do_train=False
#         model = BertForQuestionAnswering.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"), config=config)
#         model = model.to(args.device)

#         #CUDA distribute
#         if args.n_gpu > 1:                                                 
#             model = torch.nn.DataParallel(model)
#         test_dir = os.path.join(args.data_dir, 'test.jsonl')
#         test_size = len(open(test_dir,'r').readlines()) 
#         public_dataset, private_dataset = False, False
#         if test_size == 346:
#             public_dataset = True
#             print('public dataset')
#         elif test_size > 346:
#             private_dataset = True
#             print('private dataset')
#         test_convert_func = functools.partial(convert_test_data,
#                                  tokenizer=tokenizer,
#                                  max_seq_len=args.max_seq_length,
#                                  max_question_len=args.max_question_len)
#         test_examples = Test_Json_Reaser(test_dir, test_convert_func) #List[List[Test_Example]]
# #         print(len(test_examples))
#         all_test_examples = []
#         for _ in test_examples:
#             for exam in _:
#                 all_test_examples.append(exam)
# #         print(len(all_test_examples))
#         Test_dataset = TestDataset(all_test_examples)
#         logger.info('***** test_examples:' + str(len(Test_dataset))+' *****')
#         logger.info("***** Running Testing *****")
#         logger.info("  Test Batch size = %d", args.eval_batch_size)
#         # Run prediction for full data
#         test_sampler = SequentialSampler(Test_dataset)
#         Test_dataloader = DataLoader(Test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=test_collate_fn)   

#         model.eval()
#         with torch.no_grad():
#             test_result = Test_Result()
#             for inputs, examples in tqdm(Test_dataloader):#batch
#                 input_ids, input_mask, segment_ids = inputs
#                 y_preds = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device), token_type_ids=segment_ids.to(device))
#                 start_preds, end_preds, class_preds = (p.detach().cpu() for p in y_preds)
#                 start_logits, start_index = torch.max(start_preds, dim=1)
# #                                     print(start_logits.size())
#                 #(batch,)&(batch,)
#                 end_logits, end_index = torch.max(end_preds, dim=1)

#                 cls_logits = start_preds[:, 0] + end_preds[:, 0]#[cls] logits
#                 logits = start_logits+end_logits-cls_logits # (batch,)
#                 indices = torch.stack((start_index, end_index)).transpose(0,1)#(batch,2)
#                 test_result.update(examples, logits.numpy(), indices.numpy(), class_preds)
#         long_answers, short_answers, class_answers = test_result.generate_prediction()
#         print('long & short answers predict done!')
#         if public_dataset:
#             write_answer_public(long_answers, short_answers, os.path.join(args.data_dir, 'sample_submission.csv'), 'submission.csv')
#         elif private_dataset:
#             write_answer_private(long_answers, short_answers, 'submission.csv')
        
        
        
        
        
        
    
if __name__ == "__main__":
    main()    