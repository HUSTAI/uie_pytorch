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
from audioop import bias
import collections
import json
import os
import shutil
import paddle
import numpy as np

import paddle.fluid.dygraph as D
import torch
import paddle
import paddle.nn
from paddle import fluid
from paddle.utils.download import get_path_from_url

from model import UIE
from utils import logger
from pprint import pprint

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
        'encoder.embeddings.task_type_embeddings.weight':"task_type_embeddings.weight",
        'encoder.embeddings.layer_norm.weight': 'encoder.embeddings.LayerNorm.weight',
        'encoder.embeddings.layer_norm.bias': 'encoder.embeddings.LayerNorm.bias',
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
        weight_map[f'encoder.encoder.layers.{i}.norm1.weight'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.weight'
        weight_map[f'encoder.encoder.layers.{i}.norm1.bias'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.bias'
        weight_map[f'encoder.encoder.layers.{i}.linear1.weight'] = f'encoder.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.linear1.bias'] = f'encoder.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.linear2.weight'] = f'encoder.encoder.layer.{i}.output.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.linear2.bias'] = f'encoder.encoder.layer.{i}.output.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.norm2.weight'] = f'encoder.encoder.layer.{i}.output.LayerNorm.weight'
        weight_map[f'encoder.encoder.layers.{i}.norm2.bias'] = f'encoder.encoder.layer.{i}.output.LayerNorm.bias'
    # add pooler
    weight_map.update(
        {
            'encoder.pooler.dense.weight': 'encoder.pooler.dense.weight',
            'encoder.pooler.dense.bias': 'encoder.pooler.dense.bias',
            'linear_start.weight': 'linear_start.weight',
            'linear_start.bias': 'linear_start.bias',
            'linear_end.weight': 'linear_end.weight',
            'linear_end.bias': 'linear_end.bias'
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
    del config['use_task_id']
    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'),
              'wt', encoding='utf-8'), indent=4)
    logger.info('=' * 20 + 'save vocab file' + '=' * 20)
    shutil.copyfile(os.path.join(input_dir, 'vocab.txt'),
                    os.path.join(output_dir, 'vocab.txt'))
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
    with fluid.dygraph.guard():
        paddle_paddle_params, _ = D.load_dygraph(
            os.path.join(input_dir, 'model_state'))
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


def validate_model(tokenizer, pt_model, pd_model: str, atol: float = 0.05):

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

    onnx_outputs_set, ref_outputs_set = set(
        torch_outputs_dict.keys()), set(paddle_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        logger.info(
            f"\t-[x] Pytorch model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}"
        )

        raise ValueError(
            "Outputs doesn't match between reference model and Pytorch converted model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        logger.info(
            f"\t-[✓] Pytorch model output names match reference model ({onnx_outputs_set})")

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
        difference = np.amax(np.abs(ref_value - pt_value))
        if not np.allclose(ref_value, pt_value, atol=atol):
            logger.info(
                f"\t\t-[x] values not close enough (difference:{difference:.5f} atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and Pytorch converted model: "
                f"Got max absolute difference of: {difference}"
            )
        else:
            logger.info(
                f"\t\t-[✓] all values close (difference:{difference:.5f} atol: {atol})")


def compare_model(paddle_model: paddle.nn.Layer, pytorch_model: torch.nn.Module):
    from transformers import BertTokenizer
    from transformers.utils import ModelOutput

    from pprint import pprint
    import torch.nn.functional as F
    uie_torch = pytorch_model
    uie_paddle = paddle_model
    tokenizer_torch = BertTokenizer.from_pretrained(args.output_model)
    tokenizer_paddle = ErnieTokenizer.from_pretrained(args.input_model)
    gen_input_type=1
    if gen_input_type == 1:
        inputs_torch = tokenizer_torch(['航母'], ["印媒所称的“印度第一艘国产航母”—“维克兰特”号"],
                                    add_special_tokens=True,
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt")
        inputs_torch['position_ids'] = []
        for input_id in inputs_torch['input_ids']:
            position_id = torch.tensor(range(input_id.shape[-1]))
            position_id = F.pad(
                position_id, (0, 512-input_id.shape[-1]))
            inputs_torch['position_ids'].append(position_id)
        inputs_torch['position_ids'] = torch.stack(inputs_torch['position_ids'])
        inputs_torch['input_ids'] = F.pad(
            inputs_torch['input_ids'], (0, 512-inputs_torch['input_ids'].shape[-1]))
        inputs_torch['token_type_ids'] = F.pad(
            inputs_torch['token_type_ids'], (0, 512-inputs_torch['token_type_ids'].shape[-1]))
        inputs_torch['attention_mask'] = F.pad(
            inputs_torch['attention_mask'], (0, 512-inputs_torch['attention_mask'].shape[-1]))

        inputs_paddle = tokenizer_paddle(
            text=['航母'],
            text_pair=["印媒所称的“印度第一艘国产航母”—“维克兰特”号"],
            stride=len('装备'),
            truncation=True,
            max_seq_len=512,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_position_ids=True)
        for key in ['token_type_ids', 'attention_mask', 'input_ids', 'position_ids']:
            ref_value = np.array(inputs_paddle[key], dtype=np.float32)
            pt_value = np.array(inputs_torch[key], dtype=np.float32)
            assert ref_value.shape == pt_value.shape
            difference = np.amax(np.abs(ref_value - pt_value))
            print(f'input {key} difference: {difference}')
        
        rename_input = {
            'token_type_ids': 'token_type_ids',
            'attention_mask': 'att_mask',
            'input_ids': 'input_ids',
            'position_ids': 'pos_ids'

        }
        print(f'{"inputs_torch":-^50}')
        pprint(inputs_torch, width=100, compact=True)
        print(f'{"inputs_paddle":-^50}')
        pprint(inputs_paddle, width=100, compact=True)
        
        inputs_paddle_encoder = {k: paddle.to_tensor(
            inputs_paddle[k]) for k in rename_input}
        inputs_paddle = {v: paddle.to_tensor(
            inputs_paddle[k]) for k, v in rename_input.items()}
    if gen_input_type==2:
        # Type 2
        batch_size = 2
        seq_length = 6
        seq_length_with_token = seq_length+2
        max_seq_length = 512
        dummy_input = [" ".join([tokenizer.unk_token])
                    * seq_length] * batch_size
        encoded_inputs = dict(tokenizer_paddle(dummy_input, pad_to_max_seq_len=True, max_seq_len=512, return_attention_mask=True,
                                        return_position_ids=True))
        inputs_paddle = {}
        for name, value in encoded_inputs.items():
            if name == "attention_mask":
                name = "att_mask"
            if name == "position_ids":
                name = "pos_ids"
            inputs_paddle[name] = paddle.to_tensor(value, dtype=paddle.int64)
        inputs_torch = {}
        for name, value in encoded_inputs.items():
            inputs_torch[name] = torch.tensor(value, dtype=torch.int64)
    

    feature_torch = {}

    def hook_torch_wrapper(name):
        def hook_torch(module, inputs, outputs):
            if isinstance(outputs, ModelOutput):
                outputs = outputs.to_tuple()
            if isinstance(outputs,tuple):
                feature_output = [t.detach().numpy() for t in outputs]
            else:
                feature_output=outputs.detach().numpy()
                
            feature_torch[name] = feature_output
            # if name=="encoder.embeddings.word_embeddings":
            #     return outputs+torch.Tensor(task_id_embeddings)
            # if 'LayerNorm' in name:
            #     assert feature_output.shape==feature_paddle[name].shape
            #     feature_output=feature_paddle[name]
            #     return torch.tensor(feature_paddle[name])
        return hook_torch
    for name, module in pytorch_model.named_modules():
        module.register_forward_hook(hook_torch_wrapper(name))

    weight_map = build_params_map()
    layer_map = {k.rsplit(".", 1)[0]: v.rsplit(".", 1)[0]
                 for k, v in weight_map.items()}
    feature_paddle = {}

    def hook_paddle_wrapper(name):
        def hook_paddle(layer, inputs, outputs):
            if isinstance(outputs,tuple):
                feature_output = [t.numpy() for t in outputs]
            else:
                feature_output=outputs.numpy()
            feature_name = layer_map[name] if name in layer_map else name
            feature_paddle[feature_name] = feature_output
        return hook_paddle
    for prefix, layer in paddle_model.named_sublayers():
        layer.register_forward_post_hook(hook_paddle_wrapper(prefix))

    outputs_paddle = {}
    outputs_torch = {}
    
    # 运行paddle模型
    output_paddle = uie_paddle(**inputs_paddle)

    start_probs_paddle, end_probs_paddle = output_paddle
    start_probs_paddle, end_probs_paddle = start_probs_paddle.numpy(), end_probs_paddle.numpy()
    outputs_paddle['start_probs'] = start_probs_paddle
    outputs_paddle['end_probs'] = end_probs_paddle
    
    # 运行torch模型
    output_torch = uie_torch(**inputs_torch)

    start_probs_torch, end_probs_torch = output_torch[0], output_torch[1]
    start_probs_torch, end_probs_torch = start_probs_torch.detach(
    ).numpy(), end_probs_torch.detach().numpy()
    outputs_torch['start_probs'] = start_probs_torch
    outputs_torch['end_probs'] = end_probs_torch
    # logger.warning(f'paddle_value:{start_probs_paddle} pt_value:{start_probs_torch}')
    

    for key in outputs_paddle.keys():
        ref_value = np.array(outputs_paddle[key])
        pt_value = np.array(outputs_torch[key])
        assert ref_value.shape == pt_value.shape
        difference = np.amax(np.abs(ref_value - pt_value))
        print(f'output {key} difference: {difference}')

    for key in sorted(set(feature_torch.keys()) & set(feature_paddle.keys())):
        ref_list=feature_paddle[key]
        pt_list=feature_torch[key]
        if len(ref_list)!=len(pt_list):
            print(f'feature {key} output num not match')
        else:
            for i,(pt_value,ref_value) in enumerate(zip(pt_list,ref_list)):
                if ref_value.shape != pt_value.shape:
                    print(f'feature {key} output {i} shape not match')
                else:
                    difference = np.amax(np.abs(ref_value - pt_value))
                    print(f'feature {key} output {i} difference: {difference}')
    # print(f'{"start_probs_torch":-^50}')
    # pprint(start_probs_torch.detach().numpy(), width=100, compact=True)
    # print(f'{"start_probs_paddle":-^50}')
    # pprint(start_probs_paddle.numpy(), width=100, compact=True)
    pass

def test_task_id_embedding(paddle_model,inputs):
    input_token=np.array(inputs, dtype=np.int64)
    embedding_layer_paddle=paddle_model.encoder.embeddings.task_type_embeddings
    embedding_weight_paddle=embedding_layer_paddle.weight.numpy()
    embedding_result_paddle=embedding_layer_paddle(paddle.to_tensor(input_token)).numpy()
    print(f'{"embedding weight paddle":-^50}')
    pprint(embedding_weight_paddle, width=100, compact=True)
    print(f'{"embedding paddle":-^50}')
    pprint(embedding_result_paddle, width=100, compact=True)
    with open('task_id_result.json','w',encoding='utf-8') as f:
        json.dump({"embedding_result_paddle":embedding_result_paddle.tolist()},f,indent=4)
    

def test_embedding(paddle_model, pytorch_model, inputs):
    input_token=np.array(inputs, dtype=np.int64)
    embedding_layer_torch=pytorch_model.encoder.embeddings.word_embeddings
    embedding_layer_paddle=paddle_model.encoder.embeddings.word_embeddings
    embedding_weight_torch=embedding_layer_torch.weight.detach().numpy()
    embedding_weight_paddle=embedding_layer_paddle.weight.numpy()
    embedding_result_torch=embedding_layer_torch(torch.tensor(input_token)).detach().numpy()
    embedding_result_paddle=embedding_layer_paddle(paddle.to_tensor(input_token)).numpy()
    print(f'{"embedding weight torch":-^50}')
    pprint(embedding_weight_torch, width=100, compact=True)
    print(f'{"embedding weight paddle":-^50}')
    pprint(embedding_weight_paddle, width=100, compact=True)
    difference = np.amax(np.abs(embedding_weight_torch - embedding_weight_paddle))
    print(f'embedding weight difference: {difference}')
    print(f'{"embedding torch":-^50}')
    pprint(embedding_result_torch, width=100, compact=True)
    print(f'{"embedding paddle":-^50}')
    pprint(embedding_result_paddle, width=100, compact=True)
    difference = np.amax(np.abs(embedding_result_torch - embedding_result_paddle))
    print(f'embedding difference: {difference}')

def test_layer_norm(paddle_model, pytorch_model, inputs):
    # print(f'{"epsilon torch":-^50}')
    # for name, module in pytorch_model.named_modules():
    #     if 'LayerNorm' in name:
    #         print(name,module.eps)
    # print(f'{"epsilon paddle":-^50}')
    # for prefix, layer in paddle_model.named_sublayers():
    #     if 'norm' in prefix:
    #         print(prefix,layer._epsilon)
        
    input_token=np.array(inputs, dtype=np.float32)
    
    layer_norm_layer_torch=pytorch_model.encoder.embeddings.LayerNorm
    layer_norm_layer_paddle=paddle_model.encoder.embeddings.layer_norm
    epsilon_torch=layer_norm_layer_torch.eps 
    epsilon_paddle=layer_norm_layer_paddle._epsilon
    layer_norm_weight_torch=layer_norm_layer_torch.weight.detach().numpy()
    layer_norm_bias_torch=layer_norm_layer_torch.bias.detach().numpy()
    layer_norm_weight_paddle=layer_norm_layer_paddle.weight.numpy()
    layer_norm_result_torch=layer_norm_layer_torch(torch.tensor(input_token)).detach().numpy()
    layer_norm_result_paddle=layer_norm_layer_paddle(paddle.to_tensor(input_token)).numpy()

    # print(f'{"layer_norm weight torch":-^50}')
    # pprint(layer_norm_weight_torch, width=100, compact=True)
    # print(f'{"layer_norm weight paddle":-^50}')
    # pprint(layer_norm_weight_paddle, width=100, compact=True)
    difference = np.amax(np.abs(layer_norm_weight_torch - layer_norm_weight_paddle))
    print(f'layer_norm weight difference: {difference}')
    print(f'{"layer_norm torch":-^50}')
    pprint(layer_norm_result_torch, width=100, compact=True)
    print(f'{"layer_norm paddle":-^50}')
    pprint(layer_norm_result_paddle, width=100, compact=True)
    difference = np.amax(np.abs(layer_norm_result_torch - layer_norm_result_paddle))
    print(f'layer_norm difference: {difference}')

def test_linear(paddle_model, pytorch_model, inputs):
    input_token=np.array(inputs, dtype=np.float32)
    linear_layer_torch=pytorch_model.encoder.encoder.layer[0].intermediate.dense
    linear_layer_paddle=paddle_model.encoder.encoder.layers[0].linear1
    linear_weight_torch=linear_layer_torch.weight.detach().numpy().transpose()
    linear_weight_paddle=linear_layer_paddle.weight.numpy()
    linear_result_torch=linear_layer_torch(torch.tensor(input_token)).detach().numpy()
    linear_result_paddle=linear_layer_paddle(paddle.to_tensor(input_token)).numpy()
    print(f'{"linear weight torch":-^50}')
    pprint(linear_weight_torch, width=100, compact=True)
    print(f'{"linear weight paddle":-^50}')
    pprint(linear_weight_paddle, width=100, compact=True)
    difference = np.amax(np.abs(linear_weight_torch - linear_weight_paddle))
    print(f'linear weight difference: {difference}')
    print(f'{"linear torch":-^50}')
    pprint(linear_result_torch, width=100, compact=True)
    print(f'{"linear paddle":-^50}')
    pprint(linear_result_paddle, width=100, compact=True)
    difference = np.amax(np.abs(linear_result_torch - linear_result_paddle))
    print(f'linear difference: {difference}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", default="uie-base", type=str,
                        help="Directory of input paddle model.\n Will auto download model [uie-base/uie-tiny]")
    parser.add_argument("--output_model", default="uie_base_pytorch", type=str,
                        help="Directory of output pytorch model")
    parser.add_argument("--no_validate_output", action="store_true",
                        help="Directory of output pytorch model")
    args = parser.parse_args()
    check_model(args.input_model)
    extract_and_convert(args.input_model, args.output_model)
    if not args.no_validate_output:
        from paddlenlp.transformers import ErnieTokenizer
        tokenizer: ErnieTokenizer = ErnieTokenizer.from_pretrained(
            args.input_model)
        model:UIE = UIE.from_pretrained(args.output_model)
        model.eval()
        from paddlenlp.taskflow.models import UIE as UIEPaddle
        paddle_model:UIEPaddle = UIEPaddle.from_pretrained(args.input_model)
        paddle_model.eval()
        # test_embedding(paddle_model, model, [[0]])
        # test_task_id_embedding(paddle_model, [[0]])
        # test_layer_norm(paddle_model,model,10000*np.random.random(size=(2, 2, 768)).astype('float32'))
        # test_linear(paddle_model,model,np.random.random(size=(2, 2, 768)).astype('float32'))
        compare_model(paddle_model, model)
        # validate_model(tokenizer, model, paddle_model)
