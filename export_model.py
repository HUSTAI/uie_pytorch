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
import os
from itertools import chain
from typing import List, Union
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import (BertTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase)

from model import UIE
from utils import logger


def validate_onnx(tokenizer: PreTrainedTokenizerBase, pt_model: PreTrainedModel, onnx_path: Union[Path, str], strict: bool = True, atol: float = 1e-05):

    # 验证模型
    from onnxruntime import InferenceSession, SessionOptions
    from transformers import AutoTokenizer

    logger.info("Validating ONNX model...")
    if strict:
        ref_inputs = tokenizer('装备', "印媒所称的“印度第一艘国产航母”—“维克兰特”号",
                               add_special_tokens=True,
                               truncation=True,
                               max_length=512,
                               return_tensors="pt")
    else:
        batch_size = 2
        seq_length = 6
        dummy_input = [" ".join([tokenizer.unk_token])
                       * seq_length] * batch_size
        ref_inputs = dict(tokenizer(dummy_input, return_tensors="pt"))
    # ref_inputs =
    ref_outputs = pt_model(**ref_inputs)
    ref_outputs_dict = {}

    # We flatten potential collection of outputs (i.e. past_keys) to a flat structure
    for name, value in ref_outputs.items():
        # Overwriting the output name as "present" since it is the name used for the ONNX outputs
        # ("past_key_values" being taken for the ONNX inputs)
        if name == "past_key_values":
            name = "present"
        ref_outputs_dict[name] = value

    # Create ONNX Runtime session
    options = SessionOptions()
    session = InferenceSession(str(onnx_path), options, providers=[
                               "CPUExecutionProvider"])

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = {}
    for name, value in ref_inputs.items():
        onnx_inputs[name] = value.numpy()
    onnx_named_outputs = ['start_prob', 'end_prob']
    # Compute outputs from the ONNX model
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    ref_outputs_set, onnx_outputs_set = set(
        ref_outputs_dict.keys()), set(onnx_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        logger.info(
            f"\t-[x] ONNX model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}"
        )

        raise ValueError(
            "Outputs doesn't match between reference model and ONNX exported model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        logger.info(
            f"\t-[✓] ONNX model output names match reference model ({onnx_outputs_set})")

    # Check the shape and values match
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        ref_value = ref_outputs_dict[name].detach().numpy()

        logger.info(f'\t- Validating ONNX Model output "{name}":')

        # Shape
        if not ort_value.shape == ref_value.shape:
            logger.info(
                f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            raise ValueError(
                "Outputs shape doesn't match between reference model and ONNX exported model: "
                f"Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)"
            )
        else:
            logger.info(
                f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        # Values
        if not np.allclose(ref_value, ort_value, atol=atol):
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and ONNX exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")


def export_onnx(args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, device: torch.device, input_names: List[str], output_names: List[str]):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        model.config.return_dict = True
        model.config.use_cache = False

        # Create folder
        if not args.output_path.exists():
            args.output_path.mkdir(parents=True)
        save_path = args.output_path / "inference.onnx"

        dynamic_axes = {name: {0: 'batch', 1: 'sequence'}
                        for name in chain(input_names, output_names)}

        # Generate dummy input
        batch_size = 2
        seq_length = 6
        dummy_input = [" ".join([tokenizer.unk_token])
                       * seq_length] * batch_size
        inputs = dict(tokenizer(dummy_input, return_tensors="pt"))

        if save_path.exists():
            logger.warning(f'Overwrite model {save_path.as_posix()}')
            save_path.unlink()

        torch.onnx.export(model,
                          (inputs,),
                          save_path,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          opset_version=11
                          )

    if not os.path.exists(save_path):
        logger.error(f'Export Failed!')

    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=Path, required=True,
                        default='./checkpoint/model_best', help="The path to model parameters to be loaded.")
    parser.add_argument("-o", "--output_path", type=Path, default=None,
                        help="The path of model parameter in static graph to be saved.")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.model_path

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = UIE.from_pretrained(args.model_path)
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

    logger.info("Export Tokenizer Config...")

    export_tokenizer(args)

    logger.info("Export ONNX Model...")

    save_path = export_onnx(
        args, tokenizer, model, device, input_names, output_names)
    validate_onnx(tokenizer, model, save_path)

    logger.info(f"All good, model saved at: {save_path.as_posix()}")


def export_tokenizer(args):
    for tokenizer_fine in ['tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt']:
        file_from = args.model_path / tokenizer_fine
        file_to = args.output_path/tokenizer_fine
        if file_from.resolve() == file_to.resolve():
            continue
        shutil.copyfile(file_from, file_to)


if __name__ == "__main__":

    main()
