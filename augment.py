import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from multiprocessing import cpu_count
from pathlib import Path
import time
from typing import List, Tuple
import jieba
from utils import logger, tqdm, logging_redirect_tqdm

import random
from random import shuffle
import uuid


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


def eda(sentence, stop_words, synonym_replacement_ratio=0.1, random_insertion_ratio=0.1, random_swap_ratio=0.1, random_deletion_ratio=0.1, num_aug=9):
    """
    EDA函数
    https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    """
    words = jieba.lcut(sentence)
    assert len(sentence) == len("".join(words))
    words_with_uuid = [(w, uuid.uuid4()) for w in words]

    num_words = len(words_with_uuid)

    augmented_words = []
    num_new_per_technique = int(num_aug/4)+1
    n_sr = max(1, int(synonym_replacement_ratio * num_words))
    n_ri = max(1, int(random_insertion_ratio * num_words))
    n_rs = max(1, int(random_swap_ratio * num_words))

    #print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words_with_uuid, n_sr, stop_words)
        augmented_words.append(a_words)

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words_with_uuid, n_ri)
        augmented_words.append(a_words)

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words_with_uuid, n_rs)
        augmented_words.append(a_words)

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


def main():
    load_time = time.time()
    import synonyms  # 延迟加载，加载太慢了
    load_time = time.time()-load_time
    logger.info(f"synonyms loaded! time usage: {load_time:.2f}s")

    set_seed(1000)

    with open(args.doccano_file, 'r', encoding='utf-8') as f:
        doccano_data = f.readlines()

    with open(args.stopwords_file, 'r', encoding='utf-8') as f:
        stop_words = [word[:-1] for word in f.readlines()]

    new_data = []
    executor = ThreadPoolExecutor(max_workers=cpu_count()//2)
    futures = [executor.submit(process_line, stop_words, line)
               for line in doccano_data]
    pbar = tqdm(total=len(doccano_data))
    for future in as_completed(futures):
        result = future.result()
        new_data.extend(result)
        pbar.update(1)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.writelines(new_data)


def process_line(stop_words, line):
    new_data = []
    line_data = json.loads(line)
    line_text = line_data['text']
    line_labels = line_data['label']
    if not args.no_origin:
        new_data.append(line)  # 先添加原句
    augmented_words, words_with_uuid = eda(
        line_text, stop_words, num_aug=args.num_augment)
    labels_with_uuid = []
    try:
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
    except AssertionError:
        return new_data

    aug_counter = args.num_augment
    for aug_words_with_uuid in augmented_words:
        aug_counter -= 1
        if aug_counter < 0:
            break
        aug_sentence = "".join([w[0] for w in aug_words_with_uuid])
        if aug_sentence == line_text:
            continue
        aug_labels = []
        if args.verbose > 0:
            with logging_redirect_tqdm([logger.logger]):
                logger.info(f"Aug sentence: {aug_sentence}")
        try:
            for start_uuid, start_offset, end_uuid, end_offset, label in labels_with_uuid:
                start_idx = label_index(
                    aug_words_with_uuid, start_uuid, start_offset)
                end_idx = label_index(
                    aug_words_with_uuid, end_uuid, end_offset)
                assert end_idx > start_idx
                aug_labels.append([start_idx, end_idx, label])
                if args.verbose > 0:
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info(
                            f"Label: {aug_sentence[start_idx:end_idx]}")
            new_data.append(json.dumps(
                {"text": aug_sentence, "label": aug_labels}, ensure_ascii=False)+'\n')
        except:
            if args.verbose > 0:
                with logging_redirect_tqdm([logger.logger]):
                    logger.info(f"Skip augment sentence: {aug_sentence}")
            continue
    return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--doccano_file", default=Path("data/军事论坛数据集/doccano_ext.json"),
                        type=Path, help="The doccano file exported from doccano platform.")
    parser.add_argument("-o", "--output", default=Path("data/军事论坛数据集/doccano_ext_aug.json"),
                        type=Path, help="The path of data that you wanna save.")
    parser.add_argument("-s", "--stopwords_file", default=Path("./augment/stopwords/hit_stopwords.txt"),
                        type=Path, help="The path of stop words file.")
    parser.add_argument("-n", "--num_augment", type=int, default=9,
                        help="Number of augment sentence per origin sentence, defaults to 9.")
    parser.add_argument("--no_origin", action="store_true",
                        help="No original sentence.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Verbose level, defaults to 0. -vvv means 3.")
    args = parser.parse_args()
    main()
