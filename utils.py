# Copyright (c) 2022 Heiheiyoyo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import functools
import json
import logging
import math
import random
import re
import shutil
import threading
import time
from functools import partial

import colorlog
import numpy as np
import torch
from colorama import Back, Fore
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

loggers = {}

log_config = {
    'DEBUG': {
        'level': 10,
        'color': 'purple'
    },
    'INFO': {
        'level': 20,
        'color': 'green'
    },
    'TRAIN': {
        'level': 21,
        'color': 'cyan'
    },
    'EVAL': {
        'level': 22,
        'color': 'blue'
    },
    'WARNING': {
        'level': 30,
        'color': 'yellow'
    },
    'ERROR': {
        'level': 40,
        'color': 'red'
    },
    'CRITICAL': {
        'level': 50,
        'color': 'bold_red'
    }
}


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.

    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


class SpanEvaluator:
    """
    SpanEvaluator computes the precision, recall and F1-score for span detection.
    """

    def __init__(self):
        super(SpanEvaluator, self).__init__()
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def compute(self, start_probs, end_probs, gold_start_ids, gold_end_ids):
        """
        Computes the precision, recall and F1-score for span detection.
        """
        pred_start_ids = get_bool_ids_greater_than(start_probs)
        pred_end_ids = get_bool_ids_greater_than(end_probs)
        gold_start_ids = get_bool_ids_greater_than(gold_start_ids.tolist())
        gold_end_ids = get_bool_ids_greater_than(gold_end_ids.tolist())
        num_correct_spans = 0
        num_infer_spans = 0
        num_label_spans = 0
        for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
                pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids):
            [_correct, _infer, _label] = self.eval_span(
                predict_start_ids, predict_end_ids, label_start_ids,
                label_end_ids)
            num_correct_spans += _correct
            num_infer_spans += _infer
            num_label_spans += _label
        return num_correct_spans, num_infer_spans, num_label_spans

    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans

    def eval_span(self, predict_start_ids, predict_end_ids, label_start_ids,
                  label_end_ids):
        """
        evaluate position extraction (start, end)
        return num_correct, num_infer, num_label
        input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
        output: (1, 2, 2)
        """
        pred_set = get_span(predict_start_ids, predict_end_ids)
        label_set = get_span(label_start_ids, label_end_ids)
        num_correct = len(pred_set & label_set)
        num_infer = len(pred_set)
        num_label = len(label_set)
        return (num_correct, num_infer, num_label)

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(self.num_correct_spans /
                          self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(self.num_correct_spans /
                       self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(2 * precision * recall /
                         (precision + recall)) if self.num_correct_spans else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"


class IEDataset(Dataset):
    """
    Dataset for Information Extraction fron jsonl file.
    The line type is 
    {
        content
        result_list
        prompt
    }
    """

    def __init__(self, file_path, tokenizer, max_seq_len) -> None:
        super().__init__()
        self.file_path = file_path
        self.dataset = list(reader(file_path))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return convert_example(self.dataset[index], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)


def reader(data_path, max_seq_len=512):
    """
    read json
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line['content']
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result['start'] + 1 <= max_content_len < result[
                                'end']:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [
                                    result for result in result_list
                                ]
                                break
                        else:
                            break

                    json_line = {
                        'content': cur_content,
                        'result_list': cur_result_list,
                        'prompt': prompt
                    }
                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {
                            'content': res_content,
                            'result_list': result_list,
                            'prompt': prompt
                        }
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def convert_example(example, tokenizer, max_seq_len):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
        return_offsets_mapping=True)
    # encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"][0]]
    bias = 0
    for index in range(len(offset_mapping)):
        if index == 0:
            continue
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = index
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0 for x in range(max_seq_len)]
    end_ids = [0 for x in range(max_seq_len)]
    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    tokenized_output = [
        encoded_inputs["input_ids"][0], encoded_inputs["token_type_ids"][0],
        encoded_inputs["attention_mask"][0],
        start_ids, end_ids
    ]
    tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
    tokenized_output = [
        np.pad(x, (0, max_seq_len-x.shape[-1]), 'constant') for x in tokenized_output]
    return tuple(tokenized_output)


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Logger(object):
    '''
    Deafult logger in UIE

    Args:
        name(str) : Logger name, default is 'UIE'
    '''

    def __init__(self, name: str = None):
        name = 'UIE' if not name else name
        self.logger = logging.getLogger(name)

        for key, conf in log_config.items():
            logging.addLevelName(conf['level'], key)
            self.__dict__[key] = functools.partial(
                self.__call__, conf['level'])
            self.__dict__[key.lower()] = functools.partial(
                self.__call__, conf['level'])

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s',
            log_colors={key: conf['color']
                        for key, conf in log_config.items()})

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logLevel = 'DEBUG'
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._is_enable = True

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    def __call__(self, log_level: str, msg: str):
        if not self.is_enable:
            return

        self.logger.log(log_level, msg)

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator

    @contextlib.contextmanager
    def processing(self, msg: str, interval: float = 0.1):
        '''
        Continuously print a progress bar with rotating special effects.

        Args:
            msg(str): Message to be printed.
            interval(float): Rotation interval. Default to 0.1.
        '''
        end = False

        def _printer():
            index = 0
            flags = ['\\', '|', '/', '-']
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator('\r'):
                    self.info('{}: {}'.format(msg, flag))
                time.sleep(interval)
                index += 1

        t = threading.Thread(target=_printer)
        t.start()
        yield
        end = True


logger = Logger()


BAR_FORMAT = f'{{desc}}: {Fore.GREEN}{{percentage:3.0f}}%{Fore.RESET} {Fore.BLUE}{{bar}}{Fore.RESET}  {Fore.GREEN}{{n_fmt}}/{{total_fmt}} {Fore.RED}{{rate_fmt}}{{postfix}}{Fore.RESET} eta {Fore.CYAN}{{remaining}}{Fore.RESET}'
BAR_FORMAT_NO_TIME = f'{{desc}}: {Fore.GREEN}{{percentage:3.0f}}%{Fore.RESET} {Fore.BLUE}{{bar}}{Fore.RESET}  {Fore.GREEN}{{n_fmt}}/{{total_fmt}}{Fore.RESET}'
BAR_TYPE = [
    "░▝▗▖▘▚▞▛▙█",
    "░▖▘▝▗▚▞█",
    " ▖▘▝▗▚▞█",
    "░▒█",
    " >=",
    " ▏▎▍▌▋▊▉█"
    "░▏▎▍▌▋▊▉█"
]

tqdm = partial(tqdm, bar_format=BAR_FORMAT, ascii=BAR_TYPE[0], leave=False)


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= (prompt_length + 1)
        offset_map[i][1] -= (prompt_length + 1)

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(start[1] * end[1])
        sentence_id.append(
            (offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob


def cut_chinese_sent(para, rstrip=True, split_on_comma=False):
    """
    Cut the Chinese sentences more precisely, reference to 
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    flag = chr(0x1F6A9)
    if split_on_comma:
        para = re.sub(r'([，])([^”’])', rf'\1{flag}\2', para)
    para = re.sub(r'([。！？\?])([^”’])', rf'\1{flag}\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1{flag\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1{flag\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1{flag\2', para)
    if rstrip:
        para = para.rstrip()
    return para.split(f"{flag}")


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_dir='checkpoint/early_stopping', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint/early_stopping'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_pretrained(self.save_dir)
        self.val_loss_min = val_loss


def convert_cls_examples(raw_examples, prompt_prefix, options):
    examples = []
    logger.info(f"Converting doccano data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            # Compatible with doccano >= 1.6.2
            if "data" in items.keys():
                text, labels = items["data"], items["label"]
            else:
                text, labels = items["text"], items["label"]
            random.shuffle(options)
            prompt = ""
            sep = ","
            for option in options:
                prompt += option
                prompt += sep
            prompt = prompt_prefix + "[" + prompt.rstrip(sep) + "]"

            result_list = []
            example = {
                "content": text,
                "result_list": result_list,
                "prompt": prompt
            }
            for label in labels:
                start = prompt.rfind(label[0]) - len(prompt) - 1
                end = start + len(label)
                result = {"text": label, "start": start, "end": end}
                example["result_list"].append(result)
            examples.append(example)
    return examples


def add_negative_example(examples, texts, prompts, label_set, negative_ratio):
    negative_examples = []
    positive_examples = []
    with tqdm(total=len(prompts)) as pbar:
        for i, prompt in enumerate(prompts):
            negative_sample = []
            redundants_list = list(set(label_set) ^ set(prompt))
            redundants_list.sort()

            num_positive = len(examples[i])
            if num_positive != 0:
                actual_ratio = math.ceil(len(redundants_list) / num_positive)
            else:
                # Set num_positive to 1 for text without positive example
                num_positive, actual_ratio = 1, 0

            if actual_ratio <= negative_ratio or negative_ratio == -1:
                idxs = [k for k in range(len(redundants_list))]
            else:
                idxs = random.sample(
                    range(0, len(redundants_list)),
                    negative_ratio * num_positive)

            for idx in idxs:
                negative_result = {
                    "content": texts[i],
                    "result_list": [],
                    "prompt": redundants_list[idx]
                }
                negative_examples.append(negative_result)
            positive_examples.extend(examples[i])
            pbar.update(1)
    return positive_examples, negative_examples


def add_full_negative_example(examples, texts, relation_prompts, predicate_set,
                              subject_goldens):
    with tqdm(total=len(relation_prompts)) as pbar:
        for i, relation_prompt in enumerate(relation_prompts):
            negative_sample = []
            for subject in subject_goldens[i]:
                for predicate in predicate_set:
                    # The relation prompt is constructed as follows:
                    # subject + "的" + predicate
                    prompt = subject + "的" + predicate
                    if prompt not in relation_prompt:
                        negative_result = {
                            "content": texts[i],
                            "result_list": [],
                            "prompt": prompt
                        }
                        negative_sample.append(negative_result)
            examples[i].extend(negative_sample)
            pbar.update(1)
    return examples


def construct_relation_prompt_set(entity_name_set, predicate_set):
    relation_prompt_set = set()
    for entity_name in entity_name_set:
        for predicate in predicate_set:
            # The relation prompt is constructed as follows:
            # subject + "的" + predicate
            relation_prompt = entity_name + "的" + predicate
            relation_prompt_set.add(relation_prompt)
    return sorted(list(relation_prompt_set))


def convert_ext_examples(raw_examples, negative_ratio, is_train=True):
    texts = []
    entity_examples = []
    relation_examples = []
    entity_prompts = []
    relation_prompts = []
    entity_label_set = []
    entity_name_set = []
    predicate_set = []
    subject_goldens = []

    logger.info(f"Converting doccano data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            entity_id = 0
            if "data" in items.keys():
                relation_mode = False
                if isinstance(items["label"],
                              dict) and "entities" in items["label"].keys():
                    relation_mode = True
                text = items["data"]
                entities = []
                if not relation_mode:
                    # Export file in JSONL format which doccano < 1.7.0
                    for item in items["label"]:
                        entity = {
                            "id": entity_id,
                            "start_offset": item[0],
                            "end_offset": item[1],
                            "label": item[2]
                        }
                        entities.append(entity)
                        entity_id += 1
                else:
                    # Export file in JSONL format for relation labeling task which doccano < 1.7.0
                    for item in items["label"]["entities"]:
                        entity = {
                            "id": entity_id,
                            "start_offset": item["start_offset"],
                            "end_offset": item["end_offset"],
                            "label": item["label"]
                        }
                        entities.append(entity)
                        entity_id += 1
                relations = []
            else:
                # Export file in JSONL format which doccano >= 1.7.0
                if "label" in items.keys():
                    text = items["text"]
                    entities = []
                    for item in items["label"]:
                        entity = {
                            "id": entity_id,
                            "start_offset": item[0],
                            "end_offset": item[1],
                            "label": item[2]
                        }
                        entities.append(entity)
                        entity_id += 1
                    relations = []
                else:
                    # Export file in JSONL (relation) format
                    text, relations, entities = items["text"], items[
                        "relations"], items["entities"]
            texts.append(text)

            entity_example = []
            entity_prompt = []
            entity_example_map = {}
            entity_map = {}  # id to entity name
            for entity in entities:
                entity_name = text[entity["start_offset"]:entity["end_offset"]]
                entity_map[entity["id"]] = {
                    "name": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }

                entity_label = entity["label"]
                result = {
                    "text": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }
                if entity_label not in entity_example_map.keys():
                    entity_example_map[entity_label] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": entity_label
                    }
                else:
                    entity_example_map[entity_label]["result_list"].append(
                        result)

                if entity_label not in entity_label_set:
                    entity_label_set.append(entity_label)
                if entity_name not in entity_name_set:
                    entity_name_set.append(entity_name)
                entity_prompt.append(entity_label)

            for v in entity_example_map.values():
                entity_example.append(v)

            entity_examples.append(entity_example)
            entity_prompts.append(entity_prompt)

            subject_golden = []
            relation_example = []
            relation_prompt = []
            relation_example_map = {}
            for relation in relations:
                predicate = relation["type"]
                subject_id = relation["from_id"]
                object_id = relation["to_id"]
                # The relation prompt is constructed as follows:
                # subject + "的" + predicate
                prompt = entity_map[subject_id]["name"] + "的" + predicate
                if entity_map[subject_id]["name"] not in subject_golden:
                    subject_golden.append(entity_map[subject_id]["name"])
                result = {
                    "text": entity_map[object_id]["name"],
                    "start": entity_map[object_id]["start"],
                    "end": entity_map[object_id]["end"]
                }
                if prompt not in relation_example_map.keys():
                    relation_example_map[prompt] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": prompt
                    }
                else:
                    relation_example_map[prompt]["result_list"].append(result)

                if predicate not in predicate_set:
                    predicate_set.append(predicate)
                relation_prompt.append(prompt)

            for v in relation_example_map.values():
                relation_example.append(v)

            relation_examples.append(relation_example)
            relation_prompts.append(relation_prompt)
            subject_goldens.append(subject_golden)
            pbar.update(1)

    def concat_examples(positive_examples, negative_examples, negative_ratio):
        examples = []
        if math.ceil(len(negative_examples) /
                     len(positive_examples)) <= negative_ratio:
            examples = positive_examples + negative_examples
        else:
            # Random sampling the negative examples to ensure overall negative ratio unchanged.
            idxs = random.sample(
                range(0, len(negative_examples)),
                negative_ratio * len(positive_examples))
            negative_examples_sampled = []
            for idx in idxs:
                negative_examples_sampled.append(negative_examples[idx])
            examples = positive_examples + negative_examples_sampled
        return examples

    logger.info(f"Adding negative samples for first stage prompt...")
    positive_examples, negative_examples = add_negative_example(
        entity_examples, texts, entity_prompts, entity_label_set,
        negative_ratio)
    if len(positive_examples) == 0:
        all_entity_examples = []
    elif is_train:
        all_entity_examples = concat_examples(positive_examples,
                                              negative_examples, negative_ratio)
    else:
        all_entity_examples = positive_examples + negative_examples

    all_relation_examples = []
    if len(predicate_set) != 0:
        if is_train:
            logger.info(f"Adding negative samples for second stage prompt...")
            relation_prompt_set = construct_relation_prompt_set(entity_name_set,
                                                                predicate_set)
            positive_examples, negative_examples = add_negative_example(
                relation_examples, texts, relation_prompts, relation_prompt_set,
                negative_ratio)
            all_relation_examples = concat_examples(
                positive_examples, negative_examples, negative_ratio)
        else:
            logger.info(f"Adding negative samples for second stage prompt...")
            relation_examples = add_full_negative_example(
                relation_examples, texts, relation_prompts, predicate_set,
                subject_goldens)
            all_relation_examples = [
                r
                for r in relation_example
                for relation_example in relation_examples
            ]
    return all_entity_examples, all_relation_examples


def get_path_from_url(url,
                      root_dir,
                      check_exist=True,
                      decompress=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        decompress (bool): decompress zip or tar file. Default is `True`

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    import os.path
    import os
    import tarfile
    import zipfile

    def is_url(path):
        """
        Whether path is URL.
        Args:
            path (string): URL string or not.
        """
        return path.startswith('http://') or path.startswith('https://')

    def _map_path(url, root_dir):
        # parse path after download under root_dir
        fname = os.path.split(url)[-1]
        fpath = fname
        return os.path.join(root_dir, fpath)

    def _get_download(url, fullname):
        import requests
        # using requests.get method
        fname = os.path.basename(fullname)
        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.info("Downloading {} from {} failed with exception {}".format(
                fname, url, str(e)))
            return False

        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024, unit='KB') as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

        return fullname

    def _download(url, path):
        """
        Download from url, save to path.

        url (str): download url
        path (str): download to given path
        """

        if not os.path.exists(path):
            os.makedirs(path)

        fname = os.path.split(url)[-1]
        fullname = os.path.join(path, fname)
        retry_cnt = 0

        logger.info("Downloading {} from {}".format(fname, url))
        DOWNLOAD_RETRY_LIMIT = 3
        while not os.path.exists(fullname):
            if retry_cnt < DOWNLOAD_RETRY_LIMIT:
                retry_cnt += 1
            else:
                raise RuntimeError("Download from {} failed. "
                                   "Retry limit reached".format(url))

            if not _get_download(url, fullname):
                time.sleep(1)
                continue

        return fullname

    def _uncompress_file_zip(filepath):
        with zipfile.ZipFile(filepath, 'r') as files:
            file_list = files.namelist()

            file_dir = os.path.dirname(filepath)

            if _is_a_single_file(file_list):
                rootpath = file_list[0]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)

            elif _is_a_single_dir(file_list):
                # `strip(os.sep)` to remove `os.sep` in the tail of path
                rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                    os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)

                files.extractall(file_dir)
            else:
                rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                if not os.path.exists(uncompressed_path):
                    os.makedirs(uncompressed_path)
                files.extractall(os.path.join(file_dir, rootpath))

            return uncompressed_path

    def _is_a_single_file(file_list):
        if len(file_list) == 1 and file_list[0].find(os.sep) < 0:
            return True
        return False

    def _is_a_single_dir(file_list):
        new_file_list = []
        for file_path in file_list:
            if '/' in file_path:
                file_path = file_path.replace('/', os.sep)
            elif '\\' in file_path:
                file_path = file_path.replace('\\', os.sep)
            new_file_list.append(file_path)

        file_name = new_file_list[0].split(os.sep)[0]
        for i in range(1, len(new_file_list)):
            if file_name != new_file_list[i].split(os.sep)[0]:
                return False
        return True

    def _uncompress_file_tar(filepath, mode="r:*"):
        with tarfile.open(filepath, mode) as files:
            file_list = files.getnames()

            file_dir = os.path.dirname(filepath)

            if _is_a_single_file(file_list):
                rootpath = file_list[0]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)
            elif _is_a_single_dir(file_list):
                rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                    os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)
            else:
                rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                if not os.path.exists(uncompressed_path):
                    os.makedirs(uncompressed_path)

                files.extractall(os.path.join(file_dir, rootpath))

            return uncompressed_path

    def _decompress(fname):
        """
        Decompress for zip and tar file
        """
        logger.info("Decompressing {}...".format(fname))

        # For protecting decompressing interupted,
        # decompress to fpath_tmp directory firstly, if decompress
        # successed, move decompress files to fpath and delete
        # fpath_tmp and remove download compress file.

        if tarfile.is_tarfile(fname):
            uncompressed_path = _uncompress_file_tar(fname)
        elif zipfile.is_zipfile(fname):
            uncompressed_path = _uncompress_file_zip(fname)
        else:
            raise TypeError("Unsupport compress file type {}".format(fname))

        return uncompressed_path

    assert is_url(url), "downloading from {} not a url".format(url)
    fullpath = _map_path(url, root_dir)
    if os.path.exists(fullpath) and check_exist:
        logger.info("Found {}".format(fullpath))
    else:
        fullpath = _download(url, root_dir)

    if decompress and (tarfile.is_tarfile(fullpath) or
                       zipfile.is_zipfile(fullpath)):
        fullpath = _decompress(fullpath)

    return fullpath


def examples_cut_sentence(examples, split_on_comma=False):
    """
    Split doccano examples content to sentences
    """

    def _sentence_start_index(sentences, idx):
        new_idx = idx
        sent_idx = 0
        for sent in sentences:
            sent_len = len(sent)
            if new_idx < sent_len:
                return sent_idx, new_idx
            else:
                new_idx -= sent_len
                sent_idx += 1
        # return sent_idx, new_idx
        raise

    def _sentence_end_index(sentences, idx):
        new_idx = idx
        sent_idx = 0
        for sent in sentences:
            sent_len = len(sent)
            if new_idx <= sent_len:
                return sent_idx, new_idx
            else:
                new_idx -= sent_len
                sent_idx += 1
        # return sent_idx, new_idx
        raise

    examples_split = []
    for line in examples:
        items = json.loads(line)
        if "data" in items.keys():
            relation_mode = False
            if isinstance(items["label"],
                          dict) and "entities" in items["label"].keys():
                relation_mode = True
            text = items["data"]
            text_split = cut_chinese_sent(
                text, rstrip=False, split_on_comma=split_on_comma)
            assert len("".join(text_split)) == len(text)
            entities_split = [[]]*len(text_split)
            if not relation_mode:
                # Export file in JSONL format which doccano < 1.7.0
                for item in items["label"]:
                    origin_start_offset = item[0]
                    origin_end_offset = item[1]
                    label = item[2]
                    start_sent_id, start_offset = _sentence_start_index(
                        text_split, origin_start_offset)
                    end_sent_id, end_offset = _sentence_end_index(
                        text_split, origin_end_offset)
                    if start_sent_id == end_sent_id:
                        assert text[origin_start_offset:origin_end_offset] == text_split[start_sent_id][start_offset:end_offset]
                        entities_split[start_sent_id].append(
                            [start_offset, end_offset, label])
                    else:
                        for text_id in range(start_sent_id, end_sent_id+1):
                            if text_id == start_sent_id:
                                entities_split[text_id].append(
                                    [start_offset, len(text_split[text_id]), label])
                            elif text_id == end_sent_id:
                                entities_split[text_id].append(
                                    [0, end_offset, label])
                            else:
                                entities_split[text_id].append(
                                    [0, len(text_split[text_id]), label])
                            current_text = text_split[text_id]
                            current_entity = entities_split[text_id][-1]
                            current_label = current_text[current_entity[0]:current_entity[1]]
                            logger.warning(
                                f"Label 「{text[origin_start_offset:origin_end_offset]}」 split to 「{current_label}」 in 「{current_text}」")
            else:
                # Export file in JSONL format for relation labeling task which doccano < 1.7.0
                for item in items["label"]["entities"]:
                    origin_start_offset = item["start_offset"]
                    origin_end_offset = item["end_offset"]
                    label = item["label"]
                    start_sent_id, start_offset = _sentence_start_index(
                        text_split, origin_start_offset)
                    end_sent_id, end_offset = _sentence_end_index(
                        text_split, origin_end_offset)
                    if start_sent_id == end_sent_id:
                        assert text[origin_start_offset:origin_end_offset] == text_split[start_sent_id][start_offset:end_offset]
                        entities_split[start_sent_id].append(
                            [start_offset, end_offset, label])
                    else:
                        for text_id in range(start_sent_id, end_sent_id+1):
                            if text_id == start_sent_id:
                                entities_split[text_id].append(
                                    [start_offset, len(text_split[text_id]), label])
                            elif text_id == end_sent_id:
                                entities_split[text_id].append(
                                    [0, end_offset, label])
                            else:
                                entities_split[text_id].append(
                                    [0, len(text_split[text_id]), label])
                            current_text = text_split[text_id]
                            current_entity = entities_split[text_id][-1]
                            current_label = current_text[current_entity[0]:current_entity[1]]
                            logger.warning(
                                f"Label 「{text[origin_start_offset:origin_end_offset]}」 split to 「{current_label}」 in 「{current_text}」")
                for short_text, entity in zip(text_split, entities_split):
                    if short_text_rstip:=short_text.rstrip():
                        for e in entity:
                            if e[1]>len(short_text_rstip):
                                e[1]=len(short_text_rstip)
                        examples_split.append(json.dumps(
                            {"text": short_text_rstip, "label": entity}))
        else:
            # Export file in JSONL format which doccano >= 1.7.0
            if "label" in items.keys():
                text = items["text"]
                text_split = cut_chinese_sent(
                    text, rstrip=False, split_on_comma=split_on_comma)
                assert len("".join(text_split)) == len(text)
                entities_split = [[] for _ in range(len(text_split))]
                for item in items["label"]:
                    origin_start_offset = item[0]
                    origin_end_offset = item[1]
                    label = item[2]
                    entity_name =  text[origin_start_offset:origin_end_offset]
                    start_sent_id, start_offset = _sentence_start_index(
                        text_split, origin_start_offset)
                    end_sent_id, end_offset = _sentence_end_index(
                        text_split, origin_end_offset)
                    if start_sent_id == end_sent_id:
                        assert entity_name == text_split[start_sent_id][start_offset:end_offset]
                        entities_split[start_sent_id].append(
                            [start_offset, end_offset, label])
                    else:
                        for text_id in range(start_sent_id, end_sent_id+1):
                            if text_id == start_sent_id:
                                entities_split[text_id].append(
                                    [start_offset, len(text_split[text_id]), label])
                            elif text_id == end_sent_id:
                                entities_split[text_id].append(
                                    [0, end_offset, label])
                            else:
                                entities_split[text_id].append(
                                    [0, len(text_split[text_id]), label])
                            current_text = text_split[text_id]
                            current_entity = entities_split[text_id][-1]
                            current_label = current_text[current_entity[0]:current_entity[1]]
                            logger.warning(
                                f"Label 「{entity_name}」 split to 「{current_label}」 in 「{current_text}」")
                for short_text, entity in zip(text_split, entities_split):
                    if short_text_rstip:=short_text.rstrip():
                        for e in entity:
                            if e[1]>len(short_text_rstip):
                                e[1]=len(short_text_rstip)
                        examples_split.append(json.dumps(
                            {"text": short_text_rstip, "label": entity}))
            else:
                # Export file in JSONL (relation) format
                examples_split.append(line)
    
    return examples_split
