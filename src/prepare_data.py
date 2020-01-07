# from transformers import BertTokenizer
import json
from pathlib import Path
import time
import torch
import numpy as np
import random
import pandas as pd
import re
import itertools
from pandas.io.json._json import JsonReader
from typing import Callable, Dict, List, Generator, Tuple
from multiprocessing import Pool
import os
# import fire
from collections import defaultdict




class Example(object):
    def __init__(self, example_id, candidate, annotations, question_len, tokenized_to_original_index, input_ids, start_position, end_position, class_label, doc_start, doc_end, doc_token_start):
        self.example_id = example_id
        self.candidate = candidate
        self.annotations = annotations
#         self.doc_start = doc_start
        self.question_len = question_len
        self.tokenized_to_original_index = tokenized_to_original_index
        self.input_ids = input_ids
        self.start_position = start_position
        self.end_position = end_position
        self.class_label = class_label
        self.doc_ori_start = doc_start
        self.doc_ori_end = doc_end
        self.doc_token_start = doc_token_start

        
class Test_Example(object):
    def __init__(self, example_id, candidate, question_len, tokenized_to_original_index, input_ids, doc_start, doc_end, doc_token_start):
        self.example_id = example_id
        self.candidate = candidate
#         self.annotations = annotations
#         self.doc_start = doc_start
        self.question_len = question_len
        self.tokenized_to_original_index = tokenized_to_original_index
        self.input_ids = input_ids
        self.doc_ori_start = doc_start
        self.doc_ori_end = doc_end
        self.doc_token_start = doc_token_start
#         self.class_label = class_label

def convert_test_data(
    line,
    tokenizer,
    max_seq_len,
    max_question_len
):
    """Convert dictionary data into list of training data.

    Parameters
    ----------
    line : str
        Testing data.
    tokenizer : pytorch_transformers.BertTokenizer
        Tokenizer for encoding texts into ids.
    max_seq_len : int
        Maximum input sequence length.
    max_question_len : int
        Maximum input question length.
    doc_stride : int
        When splitting up a long document into chunks, how much stride to take between chunks.
    """

#     def _find_short_range(short_answers):
#         answers = pd.DataFrame(short_answers)
#         start_min = answers['start_token'].min()
#         end_max = answers['end_token'].max()
#         return start_min, end_max
    
    #model input
    data = json.loads(line)
    doc_words = data['document_text'].split()
#     print('doc_words:', len(doc_words))
    question_tokens = tokenizer.tokenize(text=data['question_text'])[:max_question_len]
#     print(question_tokens)
    original_to_tokenized_index = []
    tokenized_to_original_index = []
    all_doc_tokens = []  # tokenized document text
    for i, word in enumerate(doc_words):
        original_to_tokenized_index.append(len(all_doc_tokens))
        if re.match(r'<.+>', word):  # remove paragraph tag
            continue
        sub_tokens = tokenizer.tokenize(word)
        for sub_token in sub_tokens:
            tokenized_to_original_index.append(i)
            all_doc_tokens.append(sub_token)
    choosed_candidates = []
    index = 0
    for _ in data['long_answer_candidates']:
        index += 1
        if _['top_level'] == True:
#             print('true')
            start_position = original_to_tokenized_index[_['start_token']]
            end_position = original_to_tokenized_index[_['end_token']]
#             print(start_position, end_position)
            choosed_candidates.append((start_position, end_position, _['start_token'], _['end_token']))
#     print('num_examples:',len(choosed_candidates))
#     if index==0:
#         print(index)
    examples = []
    max_doc_len = max_seq_len - len(question_tokens) - 3  # [CLS], [SEP], [SEP]
    for t in choosed_candidates:
        doc_start, doc_end, doc_ori_start_, doc_ori_end_ = t
        doc_tokens = all_doc_tokens[doc_start:doc_end]
        if len(doc_tokens)<=max_doc_len:
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        else:
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens[:max_doc_len] + ['[SEP]']
#         print(input_tokens)
        examples.append(
            Test_Example(
                example_id=data['example_id'],
                candidate=t,
#                 annotations=annotations,
#                 doc_start=doc_start, # all_doc_token[] :index
                question_len=len(question_tokens),
                tokenized_to_original_index=tokenized_to_original_index,
                input_ids=tokenizer.convert_tokens_to_ids(input_tokens),
                doc_start=doc_ori_start_,
                doc_end=doc_ori_end_,
                doc_token_start=doc_start
#                 class_label=label
        ))

    return examples
      
    
def Test_Json_Reaser(test_file, convert_func):
    print('start process test data.....')
    with open(test_file, 'r') as f:
        lines = f.readlines()
    with Pool(10) as p:
        obj = p.map(convert_func, lines)
    print('test dataset process done!')
    return obj
    
    
def convert_data(
    line,
    tokenizer,
    max_seq_len,
    max_question_len,
    doc_stride
):
    """Convert dictionary data into list of training data.

    Parameters
    ----------
    line : str
        Training data.
    tokenizer : pytorch_transformers.BertTokenizer
        Tokenizer for encoding texts into ids.
    max_seq_len : int
        Maximum input sequence length.
    max_question_len : int
        Maximum input question length.
    doc_stride : int
        When splitting up a long document into chunks, how much stride to take between chunks.
    """

    def _find_short_range(short_answers):
        answers = pd.DataFrame(short_answers)
        start_min = answers['start_token'].min()
        end_max = answers['end_token'].max()
        return start_min, end_max
    
    #model input
    data = json.loads(line)
    doc_words = data['document_text'].split()
    
    question_tokens = tokenizer.tokenize(text=data['question_text'])[:max_question_len]
#     print(question_tokens)
    original_to_tokenized_index = []
    tokenized_to_original_index = []
    all_doc_tokens = []  # tokenized document text
    for i, word in enumerate(doc_words):
        original_to_tokenized_index.append(len(all_doc_tokens))
        if re.match(r'<.+>', word):  # remove paragraph tag
            continue
        sub_tokens = tokenizer.tokenize(word)
        for sub_token in sub_tokens:
            tokenized_to_original_index.append(i)
            all_doc_tokens.append(sub_token)
    #annotations
    annotations = data['annotations'][0]
    if annotations['yes_no_answer'] in ['YES', 'NO']:
        class_label = annotations['yes_no_answer'].lower()
        start_position = annotations['long_answer']['start_token']
        end_position = annotations['long_answer']['end_token']
    elif annotations['short_answers']:
        class_label = 'short'
        start_position, end_position = _find_short_range(annotations['short_answers'])
    elif annotations['long_answer']['candidate_index'] != -1:
        class_label = 'long'
        start_position = annotations['long_answer']['start_token']
        end_position = annotations['long_answer']['end_token']
    else:
        class_label = 'unknown'
        start_position = -1
        end_position = -1

    if start_position != -1 and end_position != -1:
        start_position = original_to_tokenized_index[start_position]
        end_position = original_to_tokenized_index[end_position]
    #choose candidates
    choosed_candidates = []
#     index = 0
    for _ in data['long_answer_candidates']:
#         index += 1
        if _['top_level'] == True:
#             print('true')
            docu_start = original_to_tokenized_index[_['start_token']]
            docu_end = original_to_tokenized_index[_['end_token']]
            choosed_candidates.append((docu_start, docu_end, _['start_token'], _['end_token']))
    examples = []
    max_doc_len = max_seq_len - len(question_tokens) - 3  # [CLS], [SEP], [SEP]
    for t in choosed_candidates:
        doc_start, doc_end, doc_ori_start_, doc_ori_end_ = t
        doc_tokens = all_doc_tokens[doc_start:doc_end]
        if doc_start <= start_position and end_position <= doc_end:
            if len(doc_tokens)<=max_doc_len:
                input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + all_doc_tokens[doc_start:doc_end] + ['[SEP]']
                start = start_position - doc_start + len(question_tokens) + 2
                end = end_position - doc_start + len(question_tokens) + 2
            else:
                if start_position+max_doc_len>doc_end:
                    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + all_doc_tokens[start_position:doc_end] + ['[SEP]']
                    doc_start = start_position
                    end = end_position - start_position + len(question_tokens) + 2
                else:
                    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + all_doc_tokens[start_position:start_position+max_doc_len] + ['[SEP]']
                    doc_start = start_position
                    if end_position > start_position+max_doc_len:
                        end = max_doc_len + len(question_tokens) + 2
                    else:
                        end = end_position - start_position + len(question_tokens) + 2
                start = 0 + len(question_tokens) + 2
            label = class_label
            
        else:
            start, end, label = -1, -1, 'unknown'   
#             doc_tokens = all_doc_tokens[doc_start:doc_end]
            if len(doc_tokens)<=max_doc_len:
                input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
            else:
                input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens[:max_doc_len] + ['[SEP]']
        assert -1 <= start < max_seq_len, f'start position is out of range: {start}'
        assert -1 <= end < max_seq_len, f'end position is out of range: {end}'
           
        examples.append(
            Example(
                example_id=data['example_id'],
                candidate=t,
                annotations=annotations,
                question_len=len(question_tokens),
                tokenized_to_original_index=tokenized_to_original_index,
                input_ids=tokenizer.convert_tokens_to_ids(input_tokens),
                start_position=start, # input_tokens
                end_position=end,
                class_label=label,
                doc_start=doc_ori_start_,
                doc_end=doc_ori_end_,
                doc_token_start=doc_start
        ))

    return examples


class JsonChunkReader(JsonReader):
    """JsonReader provides an interface for reading in a JSON file.
    """
    
    def __init__(
        self,
        filepath_or_buffer: str,
        convert_data: Callable[[str], List[Example]],
        orient: str = None,
        typ: str = 'frame',
        dtype: bool = None,
        convert_axes: bool = None,
        convert_dates: bool = True,
        keep_default_dates: bool = True,
        numpy: bool = False,
        precise_float: bool = False,
        date_unit: str = None,
        encoding: str = None,
        lines: bool = True,
        chunksize: int = 2000,
        compression: str = None,
    ):
        super(JsonChunkReader, self).__init__(
            str(filepath_or_buffer),
            orient=orient, typ=typ, dtype=dtype,
            convert_axes=convert_axes,
            convert_dates=convert_dates,
            keep_default_dates=keep_default_dates,
            numpy=numpy, precise_float=precise_float,
            date_unit=date_unit, encoding=encoding,
            lines=lines, chunksize=chunksize,
            compression=compression
        )
        self.convert_data = convert_data
        
    def __next__(self):
        lines = list(itertools.islice(self.data, self.chunksize))
        if lines:
            with Pool(10) as p:
                obj = p.map(self.convert_data, lines)
            return obj

        self.close()
        raise StopIteration


def Split(train_file, k):
    with open(train_file, 'r') as f:
        train = f.readlines()
    print(len(train))
    index = set(range(len(train)))
    k_fold = []
    for i in range(k):
        if i == k-1:
            tmp = index
        else:
            tmp = random.sample(index, int(1.0/k*len(train)))
            index -= set(tmp)
        print('number:', len(tmp))
        k_fold.append(tmp)
    os.system('mkdir ./data/')
    os.system('mkdir ./data/fold_{}/'.format(k))
    for i in range(k):
        print('fold:',i)
        os.system('mkdir ./data/fold_{}/data_{}'.format(k, i))
        dev_index = list(k_fold[i])
        train_index = []
        for j in range(k):
            if j!=i:
                train_index += k_fold[j]
        with open('./data/fold_{}/data_{}/dev.jsonl'.format(k, i), 'w') as f:
            for _ in dev_index:
                f.write(train[_])
        with open('./data/fold_{}/data_{}/train.jsonl'.format(k, i), 'w') as f:
            for _ in train_index:
                f.write(train[_])

                
# Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])
                
                
class ScoreSummary(object):
    def __init__(self):
#         self.predicted_label = None
        self.short_span_score = None
        self.cls_token_score = None
        self.answer_type_logits = None

                
class Result(object):
    """Stores results of all test data.
    """
    
    def __init__(self, max_answer_length):
        self.examples = {}
        self.start_results = {}
        self.end_results = {}
        self.class_pred = {}
        self.class_labels = ['LONG', 'NO', 'SHORT', 'UNKNOWN', 'YES']
        self.max_answer_length = max_answer_length
        
    @staticmethod
    def is_valid_index(example: Example, index: List[int]) -> bool:
        """Return whether valid index or not.
        """
        start_index, end_index = index
        if start_index > end_index:
            return False
        if start_index <= example.question_len + 2:
            return False
        return True
    
    def top_k_indices(self, logits, n_best_size):
    #     print(logits.shape) # (512,)
        indices = np.argsort(logits[1:])+1 #从小到大的索引值
#         indices = indices[token_map[indices]!=-1]
        return indices[-n_best_size:] # 取前20个概率最大的索引
    
        
    def update(self, examples, start_preds, end_preds, class_preds):
#         class_pred = torch.max(class_pred, dim=1)[1].numpy() # (batch,)
        for i, example in enumerate(examples):#batch
            if not example.example_id in self.examples.keys():
                self.examples[example.example_id] = []
            self.examples[example.example_id].append(example)
            
            if not example.example_id in self.start_results.keys():
                self.start_results[example.example_id] = []
            self.start_results[example.example_id].append(start_preds[i])
#             print(start_preds[i].shape)
            
            if not example.example_id in self.end_results.keys():
                self.end_results[example.example_id] = []
            self.end_results[example.example_id].append(end_preds[i])
#             print(end_preds[i].shape)
            
            if not example.example_id in self.class_pred.keys():
                self.class_pred[example.example_id] = []
            self.class_pred[example.example_id].append(class_preds[i])
#             print(class_preds[i].shape)

    def _generate_predictions(self):
        """Generate predictions of each examples.
        """
        long_answers = {}
        short_answers = {}
        class_answers = {}
        for i, item in enumerate(self.start_results.keys()): # line level
            score_tuples = []
            def by_score(t):
                return -t[0]
            for j, exam_start_logit in enumerate(self.start_results[item]): # example level
                start_indexes = self.top_k_indices(exam_start_logit, 20)
                end_indexes = self.top_k_indices(self.end_results[item][j], 20)
                indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))
                indexes = indexes[(indexes[:,0]<indexes[:,1])*((indexes[:,1]-indexes[:,0])<self.max_answer_length)*(indexes[:,0]>(2+self.examples[item][j].question_len))]
#                 print(indexes.shape)
#                 i = 0
                predictions = []
                for start_index, end_index in indexes: # index level
                    summary = ScoreSummary()
                    summary.short_span_score = (exam_start_logit[start_index]+self.end_results[item][j][end_index])
                    summary.cls_token_score = (exam_start_logit[0] + self.end_results[item][j][0])
            #         print(result.answer_type_logits, result.answer_type_logits.mean())
                    summary.answer_type_logits = self.class_pred[item][j]-self.class_pred[item][j].mean()#归一化
                    t_o_s = start_index-self.examples[item][j].question_len-2+self.examples[item][j].doc_token_start
                    t_o_e = end_index-self.examples[item][j].question_len-2+self.examples[item][j].doc_token_start
                    if t_o_s >= len(self.examples[item][j].tokenized_to_original_index):
                        start_span = -1
                    else:
                        start_span = self.examples[item][j].tokenized_to_original_index[t_o_s]
                    if t_o_e >= len(self.examples[item][j].tokenized_to_original_index):
                        end_span  = -1
                    else:
                        end_span = self.examples[item][j].tokenized_to_original_index[t_o_e]     
                    
                    if start_span == -1 or end_span == -1:
                        continue
                    score = summary.short_span_score - summary.cls_token_score
                    predictions.append((score, j, summary, start_span, end_span))
                
                if predictions:
#                     print(i,j)
                    
                    score_tuple = sorted(predictions, key=by_score)[0] #取分数最大的那个span
#                     print(sorted(predictions, key=by_score))
#                     short_span = Span(start_span, end_span) # example short span
                    score_tuples.append(score_tuple)
            score, index_j, summary, start_span, end_span = sorted(score_tuples, key=by_score)[0] #example中分数最高的那个
#             print('jijij',sorted(score_tuples, key=by_score))
            short_span = (start_span, end_span)
            answer_type = self.class_labels[int(np.argmax(summary.answer_type_logits))]
            if answer_type in ['YES','NO']:
                short_answer = answer_type
                long_answer_s = self.examples[item][index_j].candidate[2]
                long_answer_e = self.examples[item][index_j].candidate[3]
                long_answer = (long_answer_s, long_answer_e)
                class_answer = answer_type
            elif answer_type == 'UNKNOWN' or score < 1.5:
                short_answer = None
                long_answer = None
                class_answer = answer_type
            elif answer_type == 'SHORT':
                short_answer = short_span
                long_answer_s = self.examples[item][index_j].candidate[2]
                long_answer_e = self.examples[item][index_j].candidate[3]
                long_answer = (long_answer_s, long_answer_e)
                class_answer = answer_type
            else:
                short_answer = None
                long_answer_s = self.examples[item][index_j].candidate[2]
                long_answer_e = self.examples[item][index_j].candidate[3]
                long_answer = (long_answer_s, long_answer_e)
                class_answer = answer_type
#                 print('last else',class_answer)
            long_answers[item] = long_answer
            short_answers[item] = short_answer
            class_answers[item] = class_answer
                    
        return (long_answers, short_answers, class_answers)

    def score(self):
        """Calculate score of all examples.
        """
        def _safe_divide(x: int, y: int) -> float:
            """Compute x / y, but return 0 if y is zero.
            """
            if y == 0:
                return 0.
            else:
                return x / y

        TP, FP, FN = 0,0,0
        long_as, short_as, class_as = self._generate_predictions()
#         print('length:',len(long_as),len(short_as),len(class_as))
        for example_id in long_as.keys():
            example = self.examples[example_id][0]
            long_pred = long_as[example_id]
            short_pred = short_as[example_id]
            class_pred = class_as[example_id]
            long_label = example.annotations['long_answer']
            if long_label['candidate_index'] == -1:
                l_label = None
                if long_pred == l_label:
                    TP += 1
                else:
                    FP += 1
            else:
                l_label = (long_label['start_token'], long_label['end_token'])
                if long_pred != None and l_label[0] == long_pred[0] and l_label[1] == long_pred[1]:
                    TP += 1
                elif long_pred != None and (l_label[0] != long_pred[0] or l_label[1] != long_pred[1]):
                    FP += 1
                elif long_pred == None:
                    FN += 1
            yes_no_label = example.annotations['yes_no_answer']
            short_labels = example.annotations['short_answers'] #[{}]
        
            if len(short_labels) == 0 and yes_no_label == 'None':  # no short answer
                s_label = None
                if short_pred == s_label:
                    TP += 1
                else:
                    FP += 1
            else:  # has short answer
                if yes_no_label != 'None': # has yes_no answer
                    if short_pred == yes_no_label:
                        TP += 1
                    elif short_pred != yes_no_label and short_pred!=None:
                        FP += 1
                    elif short_pred == None:
                        FN += 1
                elif len(short_labels) != 0: # has short span
                    if short_pred != None and short_pred not in ['YES','NO']: # short span
                        flag = 0
                        for short_label in short_labels:
                            if short_label['start_token'] == short_pred[0] and \
                               short_label['end_token'] == short_pred[1]:
                                TP += 1
                                flag = 1
                                break
                        if not flag:
                            FP += 1
                    elif short_pred == None:
                        FN += 1
                    else: # yes_no
                        FP += 1
        print(TP, FP, FN)
        precision = _safe_divide(TP, TP+FP)
        recall = _safe_divide(TP, TP+FN)
        micro_f1 = _safe_divide(2*precision*recall, precision+recall)

        return micro_f1
                
# if __name__ == '__main__':
#     fire.Fire(Split)