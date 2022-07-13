# 设置方式
# 1. Select a template
# task name 选择 SequenceLabeling
# config template 选择 Custom REST Request

# 2. Set parameters
# url: http://localhost:5001/predict
# method: POST
# Params 留空
# Headers: 设置key:Content-Type value:application/json
# Body: 设置key:text value:{{ text }}
# Sample Text: 轰-6K轰炸机是中国轰-6轰炸机的最新型号

# 3. Set a template
# Mapping Template
# [
#     {% for entity in input %}
#         {
#             "start_offset": {{ entity.start_offset }},
#             "end_offset": {{ entity.end_offset}},
#             "label": "{{ entity.label }}"
#         }{% if not loop.last %},{% endif %}
#     {% endfor %}
# ]

# 4. Set mappings
# Add from:装备 to:装备


import logging
from pathlib import Path
import time
from uie_predictor import UIEPredictor
from flask import Flask, request, jsonify
from utils import logger
from colorama import Fore

from werkzeug.serving import WSGIRequestHandler


class LoggerRequestHandler(WSGIRequestHandler):
    def log(self, type, message, *args):
        getattr(logger, type)(
            f"{Fore.BLUE}[Request]{Fore.RESET} " + message.rstrip() % args)


model_path = "./checkpoint_ccks_aug2_bbs/model_best"
model_path = Path(model_path)
if not (model_path/'inference.onnx').exists():
    logger.info("Converting PyTorch model to ONNX model...")
    from model import UIE
    import torch
    from export_model import export_onnx
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = UIE.from_pretrained(model_path)
    device = torch.device('cpu')
    input_names = [
        'input_ids',
        'token_type_ids',
        'attention_mask',
    ]
    output_names = [
        'start_prob',
        'end_prob'
    ]
    save_path = export_onnx(
        model_path, tokenizer, model, device, input_names, output_names)
    logger.info("Convert complete.")

uie = UIEPredictor(model_path, schema=['装备'], engine='onnx')

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    logger_prefix = f"{Fore.GREEN}[{predict.__name__}]{Fore.RESET} "
    # 获取text参数
    text = request.form.get("text") or request.json.get(
        "text") or request.values.get("text")

    logger.info(logger_prefix+f"Text: {text}")

    start_time = time.time()
    uie_result = uie(text)[0]
    end_time = time.time()-start_time
    logger.info(logger_prefix+f"UIE time usage: {end_time*1000:.2f}ms")

    logger.info(logger_prefix+f"UIE Results: {uie_result}")

    results = []
    if '装备' in uie_result:
        for item in uie_result['装备']:
            if item['probability'] > 0.5:
                results.append(
                    {"label": "装备", "start_offset": item["start"], "end_offset": item["end"]})

    # # 返回结果
    # results = [{"result": pred}]
    logger.info(logger_prefix+f"Results: {results}")
    return jsonify(results)


if __name__ == "__main__":
    app.run("0.0.0.0", port=5001, request_handler=LoggerRequestHandler)
