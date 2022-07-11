import argparse
from itertools import chain
import json
from pathlib import Path
from uie_predictor import UIEPredictor
from utils import reader, tqdm, logger
import time


def main():
    uie = UIEPredictor(args.model_path_prefix,
                       ['装备'], engine=args.engine, device=args.device, use_fp16=args.use_fp16)
    logger.info(f"Reading {args.test_path.as_posix()}")
    if args.doccano:
        # Doccano 格式文件
        with open(args.test_path, 'r', encoding='utf-8') as f:
            testdata_doccano = f.readlines()
        testdata = []
        for line in testdata_doccano:
            line = json.loads(line)
            line_text = line["text"]
            line_labels = line['label']
            prompts_data = {}
            for start, end, label in line_labels:
                result_data = {
                    "text": line_text[start:end],
                    "start": start,
                    "end": end
                }
                prompts_data.setdefault(label, []).append(result_data)
            for prompt, result_list in prompts_data.items():
                testdata.append({
                    "content": line_text,
                    "result_list": result_list,
                    "prompt": prompt
                })

    else:
        # 模型Dataset txt格式文件
        testdata = list(reader(args.test_path))

    num_infer_spans = 0  # 预测的span数量
    num_label_spans = 0  # 标注的span数量
    num_correct_spans = 0  # 预测正确的span数量

    num_infer_tokens = 0  # 预测的token数量
    num_label_tokens = 0  # 标注的token数量
    num_correct_tokens = 0  # 预测正确的token数量

    pred_times = []

    output = open(args.output, 'w', encoding='utf-8')
    issue_id = 0

    logger.info("Start evaluating...")

    for line_id, testline in enumerate(tqdm(testdata)):
        prompt = testline['prompt']
        content = testline['content']
        label_list = testline['result_list']

        uie.set_schema(prompt)

        start_time = time.time()
        pred_result = uie([content])[0]
        end_time = time.time()

        pred_times.append(end_time-start_time)

        pred_result = pred_result[prompt] if prompt in pred_result else []

        label_set = set((item['start'], item['end']) for item in label_list)
        pred_set = set((item['start'], item['end']) for item in pred_result)

        pred_token_set = [list(range(span[0], span[1])) for span in pred_set]
        pred_token_set = set(chain.from_iterable(pred_token_set))
        label_token_set = [list(range(span[0], span[1])) for span in label_set]
        label_token_set = set(chain.from_iterable(label_token_set))

        num_infer_spans += len(pred_set)
        num_label_spans += len(label_set)
        num_correct_spans += len(pred_set & label_set)

        num_correct_tokens += len(pred_token_set & label_token_set)
        num_infer_tokens += len(pred_token_set)
        num_label_tokens += len(label_token_set)

        if len(pred_set) == len(label_set) and len(label_set) == len(pred_set & label_set):
            continue
        output.write(
            f"{'':-^20} line_id:{line_id} issue_id:{issue_id} {'':-^20}\n")
        output.write(f"{content}\n")
        output.write(f"{'':-^20} 标注 {'':-^20}\n")
        for item in label_list:
            item_start = item['start']
            item_end = item['end']
            item_text = item['text']
            output.write(f"[{item_start}, {item_end}]，{item_text}\n")
        output.write(f"{'':-^20} 预测 {'':-^20}\n")
        for item in pred_result:
            item_start = item['start']
            item_end = item['end']
            item_text = item['text']
            probability = item['probability']
            output.write(
                f"[{item_start}, {item_end}]，{item_text}，{probability:.2%}\n")
        output.write(f"{'':-^50}\n\n")
        issue_id += 1

    precision = float(num_correct_spans /
                      num_infer_spans) if num_infer_spans else 0.
    recall = float(num_correct_spans /
                   num_label_spans) if num_label_spans else 0.
    f1_score = float(2 * precision * recall /
                     (precision + recall)) if num_correct_spans else 0.

    precision_token = float(num_correct_tokens /
                            num_infer_tokens) if num_infer_tokens else 0.
    recall_token = float(num_correct_tokens /
                         num_label_tokens) if num_label_tokens else 0.
    f1_score_token = float(2 * precision_token * recall_token /
                           (precision_token + recall_token)) if num_correct_tokens else 0.

    tp = num_correct_spans
    fp = num_infer_spans-num_correct_spans
    fn = num_label_spans-num_correct_spans
    output.write(f"{'':-^10}\n")
    output.write(f"TP: {tp} FP: {fp} FN: {fn}\n")
    output.write(
        f"precision: {precision:.2%} recall:{recall:.2%} F1: {f1_score:.2%}\n")
    logger.info(
        f"Evaluation precision: {precision:.2%} recall:{recall:.2%} F1: {f1_score:.2%}")
    output.write(
        f"Token precision: {precision_token:.2%} Token recall:{recall_token:.2%} Token F1: {f1_score_token:.2%}\n")
    logger.info(
        f"Evaluation Token precision: {precision_token:.2%} Token recall:{recall_token:.2%} Token F1: {f1_score_token:.2%}")
    time_sum = sum(pred_times)
    time_avg = time_sum/len(pred_times)
    output.write(f"平均用时: {time_avg*1000:.2f}ms 总计用时:{time_sum:.3f}s\n")
    logger.info(f"平均用时: {time_avg*1000:.2f}ms 总计用时:{time_sum:.3f}s")
    logger.info(f"Complete! Write result to {args.output.as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path_prefix", type=str,
                        default='checkpoint_ccks_bbs/model_best',
                        help="The path prefix of inference model to be used.")
    parser.add_argument("--use_fp16", action='store_true',
                        help="Whether to use fp16 inference, only takes effect when deploying on gpu.",)
    parser.add_argument("-D", "--device", choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to run model, defaults to gpu, defaults to gpu.")
    parser.add_argument("-e", "--engine", choices=['pytorch', 'onnx'], default="pytorch",
                        help="Select which engine to run model, defaults to pytorch.")
    parser.add_argument("-t", "--test_path", type=Path, default=Path("data/军事论坛数据集/test.txt"),
                        help="The path of test set.")
    parser.add_argument("-o", "--output", type=Path, default=Path('eval_result.txt'),
                        help="Path of output txt stats file.")
    parser.add_argument("-d", "--doccano", action='store_true',
                        help="Whether the test set is Doccano format.")
    args = parser.parse_args()
    main()
