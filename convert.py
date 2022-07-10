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

import argparse
import collections
import json
import os
import pickle
import shutil
import numpy as np

import torch
try:
    import paddle
    from paddle.utils.download import get_path_from_url
    paddle_installed = True
except (ImportError, ModuleNotFoundError):
    from utils import get_path_from_url
    paddle_installed = False

from model import UIE
from utils import logger

MODEL_MAP = {
    "uie-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json"
        }
    },
    "uie-medium": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-mini": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-micro": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-nano": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-medical-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-tiny": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/tokenizer_config.json"
        }
    }
}


def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'encoder.embeddings.word_embeddings.weight': "encoder.embeddings.word_embeddings.weight",
        'encoder.embeddings.position_embeddings.weight': "encoder.embeddings.position_embeddings.weight",
        'encoder.embeddings.token_type_embeddings.weight': "encoder.embeddings.token_type_embeddings.weight",
        'encoder.embeddings.task_type_embeddings.weight': "encoder.embeddings.task_type_embeddings.weight",
        'encoder.embeddings.layer_norm.weight': 'encoder.embeddings.LayerNorm.gamma',
        'encoder.embeddings.layer_norm.bias': 'encoder.embeddings.LayerNorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'encoder.encoder.layers.{i}.self_attn.q_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.q_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.k_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.k_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.v_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.v_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.out_proj.weight'] = f'encoder.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.out_proj.bias'] = f'encoder.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.norm1.weight'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'encoder.encoder.layers.{i}.norm1.bias'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'encoder.encoder.layers.{i}.linear1.weight'] = f'encoder.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.linear1.bias'] = f'encoder.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.linear2.weight'] = f'encoder.encoder.layer.{i}.output.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.linear2.bias'] = f'encoder.encoder.layer.{i}.output.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.norm2.weight'] = f'encoder.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'encoder.encoder.layers.{i}.norm2.bias'] = f'encoder.encoder.layer.{i}.output.LayerNorm.beta'
    # add pooler
    weight_map.update(
        {
            'encoder.pooler.dense.weight': 'encoder.pooler.dense.weight',
            'encoder.pooler.dense.bias': 'encoder.pooler.dense.bias',
            'linear_start.weight': 'linear_start.weight',
            'linear_start.bias': 'linear_start.bias',
            'linear_end.weight': 'linear_end.weight',
            'linear_end.bias': 'linear_end.bias',
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(
        open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8'))
    config = config['init_args'][0]
    config["architectures"] = ["UIE"]
    config['layer_norm_eps'] = 1e-12
    del config['init_class']
    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'),
              'wt', encoding='utf-8'), indent=4)
    logger.info('=' * 20 + 'save vocab file' + '=' * 20)
    with open(os.path.join(input_dir, 'vocab.txt'), 'rt', encoding='utf-8') as f:
        words = f.read().splitlines()
    words_set = set()
    words_duplicate_indices = []
    for i in range(len(words)-1, -1, -1):
        word = words[i]
        if word in words_set:
            words_duplicate_indices.append(i)
        words_set.add(word)
    for i, idx in enumerate(words_duplicate_indices):
        words[idx] = chr(0x1F6A9+i)  # Change duplicated word to 🚩 LOL
    with open(os.path.join(output_dir, 'vocab.txt'), 'wt', encoding='utf-8') as f:
        for word in words:
            f.write(word+'\n')
    special_tokens_map = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]"
    }
    json.dump(special_tokens_map, open(os.path.join(output_dir, 'special_tokens_map.json'),
              'wt', encoding='utf-8'))
    tokenizer_config = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "tokenizer_class": "BertTokenizer"
    }
    json.dump(tokenizer_config, open(os.path.join(output_dir, 'tokenizer_config.json'),
              'wt', encoding='utf-8'))
    logger.info('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    if paddle_installed:
        import paddle.fluid.dygraph as D
        from paddle import fluid
        with fluid.dygraph.guard():
            paddle_paddle_params, _ = D.load_dygraph(
                os.path.join(input_dir, 'model_state'))
    else:
        paddle_paddle_params = pickle.load(
            open(os.path.join(input_dir, 'model_state.pdparams'), 'rb'))
        del paddle_paddle_params['StructuredToParameterName@@']
    for weight_name, weight_value in paddle_paddle_params.items():
        if 'weight' in weight_name:
            if 'encoder.encoder' in weight_name or 'pooler' in weight_name or 'linear' in weight_name:
                weight_value = weight_value.transpose()
        # Fix: embedding error
        if 'word_embeddings.weight' in weight_name:
            weight_value[0, :] = 0
        if weight_name not in weight_map:
            logger.info(f"{'='*20} [SKIP] {weight_name} {'='*20}")
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        logger.info(
            f"{weight_name} -> {weight_map[weight_name]} {weight_value.shape}")
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def check_model(input_model):
    if not os.path.exists(input_model):
        if input_model not in MODEL_MAP:
            raise ValueError('input_model not exists!')

        resource_file_urls = MODEL_MAP[input_model]['resource_file_urls']
        logger.info("Downloading resource files...")

        for key, val in resource_file_urls.items():
            file_path = os.path.join(input_model, key)
            if not os.path.exists(file_path):
                get_path_from_url(val, input_model)


def validate_model(tokenizer, pt_model, pd_model: str, atol: float = 1e-5):

    logger.info("Validating PyTorch model...")

    batch_size = 2
    seq_length = 6
    seq_length_with_token = seq_length+2
    max_seq_length = 512
    dummy_input = [" ".join([tokenizer.unk_token])
                   * seq_length] * batch_size
    encoded_inputs = dict(tokenizer(dummy_input, pad_to_max_seq_len=True, max_seq_len=512, return_attention_mask=True,
                                    return_position_ids=True))
    paddle_inputs = {}
    for name, value in encoded_inputs.items():
        if name == "attention_mask":
            name = "att_mask"
        if name == "position_ids":
            name = "pos_ids"
        paddle_inputs[name] = paddle.to_tensor(value, dtype=paddle.int64)

    paddle_named_outputs = ['start_prob', 'end_prob']
    paddle_outputs = pd_model(**paddle_inputs)

    torch_inputs = {}
    for name, value in encoded_inputs.items():
        torch_inputs[name] = torch.tensor(value, dtype=torch.int64)
    torch_outputs = pt_model(**torch_inputs)
    torch_outputs_dict = {}

    for name, value in torch_outputs.items():
        torch_outputs_dict[name] = value

    torch_outputs_set, ref_outputs_set = set(
        torch_outputs_dict.keys()), set(paddle_named_outputs)
    if not torch_outputs_set.issubset(ref_outputs_set):
        logger.info(
            f"\t-[x] Pytorch model output names {torch_outputs_set} do not match reference model {ref_outputs_set}"
        )

        raise ValueError(
            "Outputs doesn't match between reference model and Pytorch converted model: "
            f"{torch_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        logger.info(
            f"\t-[✓] Pytorch model output names match reference model ({torch_outputs_set})")

    # Check the shape and values match
    for name, ref_value in zip(paddle_named_outputs, paddle_outputs):
        ref_value = ref_value.numpy()
        pt_value = torch_outputs_dict[name].detach().numpy()
        logger.info(f'\t- Validating PyTorch Model output "{name}":')

        # Shape
        if not pt_value.shape == ref_value.shape:
            logger.info(
                f"\t\t-[x] shape {pt_value.shape} doesn't match {ref_value.shape}")
            raise ValueError(
                "Outputs shape doesn't match between reference model and Pytorch converted model: "
                f"Got {ref_value.shape} (reference) and {pt_value.shape} (PyTorch)"
            )
        else:
            logger.info(
                f"\t\t-[✓] {pt_value.shape} matches {ref_value.shape}")

        # Values
        if not np.allclose(ref_value, pt_value, atol=atol):
            logger.info(
                f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and Pytorch converted model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - pt_value))}"
            )
        else:
            logger.info(
                f"\t\t-[✓] all values close (atol: {atol})")


def do_main():
    check_model(args.input_model)
    extract_and_convert(args.input_model, args.output_model)
    if not args.no_validate_output:
        if paddle_installed:
            try:
                from paddlenlp.transformers import ErnieTokenizer
                from paddlenlp.taskflow.models import UIE as UIEPaddle
            except (ImportError, ModuleNotFoundError) as e:
                raise ModuleNotFoundError(
                    'Module PaddleNLP is not installed. Try install paddlenlp or run convert.py with --no_validate_output') from e
            tokenizer: ErnieTokenizer = ErnieTokenizer.from_pretrained(
                args.input_model)
            model = UIE.from_pretrained(args.output_model)
            model.eval()
            paddle_model = UIEPaddle.from_pretrained(args.input_model)
            paddle_model.eval()
            validate_model(tokenizer, model, paddle_model)
        else:
            logger.warning("Skipping validating PyTorch model because paddle is not installed. "
                           "The outputs of the model may not be the same as Paddle model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", default="uie-base", type=str,
                        help="Directory of input paddle model.\n Will auto download model [uie-base/uie-tiny]")
    parser.add_argument("-o", "--output_model", default="uie_base_pytorch", type=str,
                        help="Directory of output pytorch model")
    parser.add_argument("--no_validate_output", action="store_true",
                        help="Directory of output pytorch model")
    args = parser.parse_args()

    do_main()
