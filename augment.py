import argparse
import json
import math
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from random import shuffle
from typing import List, Tuple
import queue

import jieba
from colorama import Fore

from utils import logger, logging_redirect_tqdm, tqdm


class EntityChoice:
    def __init__(self, entity_list: List[str]) -> None:
        self.entity_list = entity_list.copy()
        self.entity_list_queue = queue.Queue()
        entity_list_shuffle = entity_list.copy()
        random.shuffle(entity_list_shuffle)
        for e in entity_list:
            self.entity_list_queue.put_nowait(e)

    def __call__(self):
        if not self.entity_list_queue.empty():
            try:
                return self.entity_list_queue.get_nowait()
            except queue.Empty:
                return random.choice(self.entity_list)
        else:
            return random.choice(self.entity_list)


def txt_to_doccano_line(line: str):
    data = json.loads(line)
    doccano_data = {
        "text": data["content"],
        "label": [
            [d['start'], d['end'], data["prompt"]]
            for d in data["result_list"]
        ]
    }
    return json.dumps(doccano_data, ensure_ascii=False)+"\n"


def doccano_to_txt_line(line: str):
    data = json.loads(line)
    txt_data = {
        "content": data["text"],
        "result_list": [{
            "text": data["text"][start:end],
            "start":start,
            "end":end
        }for start, end, _ in data["label"]],
        "prompt": "装备"
    }
    return json.dumps(txt_data, ensure_ascii=False)+"\n"


def set_seed(seed):
    random.seed(seed)


def synonym_replacement(words_with_uuid, n, stop_words):
    """
    同义词替换
    替换一个语句中的n个单词为其同义词
    https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    """
    new_words = words_with_uuid.copy()
    random_word_list = list(
        set([word[0] for word in words_with_uuid if word[0] not in stop_words and word[0] != " "]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [(synonym, uuid.uuid4()) if word[0] ==
                         random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return new_words


def get_synonyms(word):
    import synonyms
    return synonyms.nearby(word)[0]


def random_insertion(words_with_uuid, n):
    """
    随机插入
    随机在语句中插入n个词
    https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    """
    new_words = words_with_uuid.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words_with_uuid):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words_with_uuid[random.randint(
            0, len(new_words_with_uuid)-1)]
        synonyms = get_synonyms(random_word[0])
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words_with_uuid)-1)
    new_words_with_uuid.insert(random_idx, (random_synonym, uuid.uuid4()))


def random_swap(words, n):
    """
    Random swap
    Randomly swap two words in the sentence n times
    https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    """
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    """
    https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    """
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_deletion(words, p):
    """
    随机删除
    以概率p删除语句中的词
    https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    """
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p or word[0] == " ":
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words


def eda(words, stop_words, synonym_replacement_ratio=0.1, random_insertion_ratio=0.1, random_swap_ratio=0.1, random_deletion_ratio=0.1, num_aug=9):
    """
    EDA函数
    https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    """

    words_with_uuid = [(w, uuid.uuid4()) for w in words]

    num_words = len(words_with_uuid)

    augmented_words = []
    num_new_per_technique = int(num_aug/3)+1
    n_sr = max(1, int(synonym_replacement_ratio * num_words))
    n_ri = max(1, int(random_insertion_ratio * num_words))
    # n_rs = max(1, int(random_swap_ratio * num_words))

    #print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words_with_uuid, n_sr, stop_words)
        augmented_words.append(a_words)

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words_with_uuid, n_ri)
        augmented_words.append(a_words)

    # # 随机交换rs
    # for _ in range(num_new_per_technique):
    #     a_words = random_swap(words_with_uuid, n_rs)
    #     augmented_words.append(a_words)

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words_with_uuid, random_deletion_ratio)
        augmented_words.append(a_words)

    # print(augmented_sentences)
    shuffle(augmented_words)

    # if num_aug >= 1:
    #     augmented_words = augmented_words[:num_aug]
    # else:
    #     keep_prob = num_aug / len(augmented_words)
    #     augmented_words = [
    #         s for s in augmented_words if random.uniform(0, 1) < keep_prob]

    # augmented_words.append(words_with_uuid)

    return augmented_words, words_with_uuid


def label_start_uuid(words_with_uuid: List[Tuple[str, uuid.UUID]], idx: int) -> Tuple[uuid.UUID, int]:
    """将起始索引转成UUID和偏置的形式

    Args:
        words_with_uuid (List[Tuple[str,uuid.UUID]]): 带uuid的词列表
        idx (int): 起始索引

    Raises:
        IndexError: 超出索引

    Returns:
        Tuple[uuid.UUID,int]: 起始所在的词UUID和offset
    """
    new_idx = idx
    for word, uuid_ in words_with_uuid:
        word_len = len(word)
        if new_idx < word_len:
            return uuid_, new_idx
        else:
            new_idx -= word_len
    raise IndexError


def label_index(words_with_uuid: List[Tuple[str, uuid.UUID]], word_uuid: uuid.UUID, offset: int) -> int:
    """将词UUID加offset的形式转成索引的形式

    Args:
        words_with_uuid (List[Tuple[str, uuid.UUID]]): 带uuid的词列表
        word_uuid (uuid.UUID): 所在词的UUID
        offset (int): 相对词的偏置

    Raises:
        IndexError: 超出索引或找不到UUID

    Returns:
        int: 对应的索引
    """
    new_idx = 0
    for word, uuid_ in words_with_uuid:
        if uuid_ == word_uuid:
            offset = len(word) if offset > len(word) else offset
            return new_idx+offset
        else:
            new_idx += len(word)
    raise IndexError


def label_end_uuid(words_with_uuid: List[Tuple[str, uuid.UUID]], idx: int) -> Tuple[uuid.UUID, int]:
    """将结束索引转成UUID和偏置的形式

    Args:
        words_with_uuid (List[Tuple[str, uuid.UUID]]): 带uuid的词列表
        idx (int): 结束索引

    Raises:
        IndexError: 超出索引

    Returns:
        Tuple[uuid.UUID, int]: 结束所在的词UUID和offset
    """
    new_idx = idx
    for word, uuid_ in words_with_uuid:
        word_len = len(word)
        if new_idx <= word_len:
            return uuid_, new_idx
        else:
            new_idx -= word_len
    raise IndexError


def eda_line(line: str, stop_words: List[str], num_aug: int = 9, add_origin: bool = True) -> List[str]:
    """对一行进行EDA操作

    Args:
        line (str): 一行doccano格式的数据
        stop_words (List[str]): 停用词表
        num_aug (int, optional): 新增数据量. Defaults to 9.
        add_origin (bool, optional): 添加原句. Defaults to true.

    Returns:
        List[str]: EDA操作后的新数据
    """
    line_data = json.loads(line)
    line_text = line_data['text']
    line_labels = line_data['label']

    new_data = []
    if add_origin:
        new_data.append(line)
        num_aug -= 1

    words = []
    last_end = 0
    for start, end, _ in sorted(set(tuple(item) for item in line_labels)):
        if start > last_end:
            words.extend(jieba.lcut(line_text[last_end:start]))
        words.extend(jieba.lcut(line_text[start:end]))
        last_end = end
    if len(line_text) > last_end:
        words.extend(jieba.lcut(line_text[last_end:len(line_text)]))

    # words = jieba.lcut(line_text)
    assert len(line_text) == len("".join(words))
    augmented_words, words_with_uuid = eda(
        words, stop_words, num_aug=num_aug)

    labels_with_uuid = []
    try:
        labels_with_uuid = labels_to_labels_with_uuid(
            line_text, line_labels, words_with_uuid)
    except AssertionError:
        return new_data

    aug_counter = num_aug
    for aug_words_with_uuid in augmented_words:
        aug_counter -= 1
        if aug_counter < 0:
            break
        aug_sentence = "".join([w[0] for w in aug_words_with_uuid])
        if aug_sentence == line_text:
            continue
        aug_labels = []

        try:
            aug_labels = labels_with_uuid_to_labels(
                labels_with_uuid, aug_words_with_uuid)
            new_data.append(json.dumps(
                {"text": aug_sentence, "label": aug_labels}, ensure_ascii=False)+'\n')
        except:
            if args.verbose > 0:
                with logging_redirect_tqdm([logger.logger]):
                    logger.info(f"Skip augment sentence: {aug_sentence}")
            continue
        if args.verbose > 0:
            with logging_redirect_tqdm([logger.logger]):
                sentence_to_print = aug_sentence
                aug_entitys = [aug_sentence[start:end]
                               for start, end, _ in aug_labels]
                for ae in aug_entitys:
                    sentence_to_print = sentence_to_print.replace(
                        ae, f"{Fore.GREEN}{ae}{Fore.RESET}")
                logger.info(f"Eda sentence: {sentence_to_print}")
    return new_data


def labels_with_uuid_to_labels(labels_with_uuid, aug_words_with_uuid):
    aug_labels = []
    for start_uuid, start_offset, end_uuid, end_offset, label in labels_with_uuid:
        start_idx = label_index(
            aug_words_with_uuid, start_uuid, start_offset)
        end_idx = label_index(
            aug_words_with_uuid, end_uuid, end_offset)
        assert end_idx > start_idx
        aug_labels.append([start_idx, end_idx, label])
    return aug_labels


def labels_to_labels_with_uuid(line_text, line_labels, words_with_uuid):
    labels_with_uuid = []
    for start, end, label in line_labels:
        start_uuid, start_offset = label_start_uuid(
            words_with_uuid, start)
        try:
            assert start_offset == 0
        except AssertionError:
            word_start = [w[0]
                          for w in words_with_uuid if w[1] == start_uuid][0]
            with logging_redirect_tqdm([logger.logger]):
                logger.error(
                    f'Jieba Cut Error：Label: {line_text[start:end]} start_word: {word_start}')
            raise
        end_uuid, end_offset = label_end_uuid(words_with_uuid, end)
        labels_with_uuid.append(
            (start_uuid, start_offset, end_uuid, end_offset, label))
    return labels_with_uuid


def entity_replacement(line: str, num_aug: int, entity_choice: EntityChoice, add_origin: bool = True) -> List[str]:
    """根据实体列表中的实体替换数据中的实体

    例如
    `歼15是...`
    替换为
    `辽宁舰是...`

    Args:
        line (str): 一行数据，doccano格式
        num_aug (int): 增强数据数量
        entity_list (List[str]): 实体列表
        add_origin (bool, optional): 是否添加原句. Defaults to True.

    Returns:
        List[str]: 新数据
    """
    line_data = json.loads(line)
    line_text = line_data['text']
    line_labels = line_data['label']

    new_data = []
    if add_origin:
        new_data.append(line)  # 先添加原句
        num_aug -= 1
    if not line_labels:
        # 没有标注提及
        return new_data
    for _ in range(num_aug):
        new_sentence = line_text
        entitys = list(sorted(set(tuple(item)
                       for item in line_labels)))
        new_labels = []
        offset = 0
        new_end = 0

        for start, end, label in entitys:
            old_entity = line_text[start:end]
            start = start+offset
            end = end+offset
            assert start >= new_end, "Nested entity detected!"
            entity = entity_choice()
            new_end = start+len(entity)

            new_sentence = list(new_sentence)
            new_sentence[start:end] = list(entity)
            new_sentence = ''.join(new_sentence)

            offset += new_end-end

            new_labels.append([start, new_end, label])

            if args.verbose > 0:
                with logging_redirect_tqdm([logger.logger]):
                    logger.info(
                        f"{Fore.BLUE}{old_entity}{Fore.RESET} -> {Fore.BLUE}{entity}{Fore.RESET}")

        if args.verbose > 0:
            with logging_redirect_tqdm([logger.logger]):
                sentence_to_print = new_sentence
                new_entitys = [new_sentence[start:end]
                               for start, end, _ in new_labels]
                for ne in new_entitys:
                    sentence_to_print = sentence_to_print.replace(
                        ne, f"{Fore.GREEN}{ne}{Fore.RESET}")
                logger.info(f"Replace sentence: {sentence_to_print}")

        new_data.append(json.dumps(
            {"text": new_sentence, "label": new_labels}, ensure_ascii=False
        )+"\n")

        random.shuffle(new_data)

    return new_data


def process_line(line: str, stop_words: List[str], entity_choice: EntityChoice) -> List[str]:
    """对一行进行处理，生成新的数据

    Args:
        line (str): 输入的一行doccano数据
        stop_words (List[str]): 停用词表
        entity_list (List[str]): 实体表

    Returns:
        List[str]: 新生成的数据
    """
    num_aug_eda = int(math.sqrt(args.num_augment))
    num_aug_entity = num_aug_eda+1 \
        if num_aug_eda * num_aug_eda < args.num_augment \
        else num_aug_eda
    new_data_eda = eda_line(line, stop_words,
                            num_aug_eda, add_origin=not args.no_origin)
    new_data = []
    for new_line in new_data_eda:
        new_data_entity = entity_replacement(
            new_line, num_aug_entity, entity_choice)
        new_data.extend(new_data_entity)
    if len(new_data) < args.num_augment:
        new_data.extend(eda_line(line, stop_words,
                                 args.num_augment-len(new_data),
                                 add_origin=not args.no_origin))
    random.shuffle(new_data)

    return new_data[:args.num_augment]


def main():
    logger.info("Start loading synonyms...")
    load_time = time.time()
    import synonyms  # 延迟加载，加载太慢了
    load_time = time.time()-load_time
    logger.info(f"Synonyms loaded! time usage: {load_time:.2f}s")

    set_seed(1000)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        doccano_data = f.readlines()

    if 'content' in json.loads(doccano_data[0]):
        logger.info("Converting txt format to doccano format...")
        doccano_data = [txt_to_doccano_line(line)
                        for line in doccano_data]

    with open(args.stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = [word[:-1] for word in f.readlines()]

    with open(args.entity_file, 'r', encoding='utf-8') as f:
        entity_list = [word[:-1] for word in f.readlines()]

    entity_choice = EntityChoice(entity_list)

    logger.info("Starting augment...")

    new_data = []
    if not args.single_thread:
        executor = ThreadPoolExecutor(max_workers=cpu_count()//2)
        futures = [executor.submit(process_line, line, stop_words, entity_choice)
                   for line in doccano_data]
        pbar = tqdm(total=len(doccano_data))
        for future in as_completed(futures):
            result = future.result()
            new_data.extend(result)
            pbar.update(1)
    else:
        for line in tqdm(doccano_data):
            new_data.extend(process_line(line, stop_words, entity_choice))
    if args.output_file.suffix in [".txt"]:
        logger.info("Converting to txt format...")
        new_data = [doccano_to_txt_line(line) for line in new_data]
    if not args.output_file.parent.exists():
        args.output_file.parent.mkdir()
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default=Path("data/军事论坛数据集/doccano_ext.json"),
                        type=Path, help="The doccano file exported from doccano platform.")
    parser.add_argument("-o", "--output_file", default=Path("data/军事论坛数据集/doccano_ext_aug.json"),
                        type=Path, help="The path of data that you wanna save.")
    parser.add_argument("-s", "--stopwords_file", default=Path("./augment/stopwords/hit_stopwords.txt"),
                        type=Path, help="The path of stop words file.")
    parser.add_argument("-e", "--entity_file", default=Path("./augment/entity/entity_list.txt"),
                        type=Path, help="The path of entity list file.")
    parser.add_argument("-n", "--num_augment", type=int, default=9,
                        help="Number of augment sentence per origin sentence, defaults to 9.")
    parser.add_argument("--no_origin", action="store_true",
                        help="No original sentence.")
    parser.add_argument("--single_thread", action="store_true",
                        help="Use single thread, donnot use multi-thread.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Verbose level, defaults to 0. -vvv means 3.")
    args = parser.parse_args()
    main()
