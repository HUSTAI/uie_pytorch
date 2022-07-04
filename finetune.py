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
import time
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from utils import IEDataset, logger, tqdm
from model import UIE
from evaluate import evaluate
from utils import set_seed, SpanEvaluator, EarlyStopping, logging_redirect_tqdm


def do_train():

    set_seed(args.seed)
    show_bar = True

    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    model = UIE.from_pretrained(args.model)
    if args.device == 'gpu':
        model = model.cuda()
    train_ds = IEDataset(args.train_path, tokenizer=tokenizer,
                         max_seq_len=args.max_seq_len)
    dev_ds = IEDataset(args.dev_path, tokenizer=tokenizer,
                       max_seq_len=args.max_seq_len)

    train_data_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    dev_data_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=True)

    # optimizer = paddle.optimizer.AdamW(
    #     learning_rate=args.learning_rate, parameters=model.parameters())
    optimizer = torch.optim.AdamW(
        lr=args.learning_rate, params=model.parameters())

    # criterion = paddle.nn.BCELoss()
    criterion = torch.nn.functional.binary_cross_entropy
    metric = SpanEvaluator()

    if args.early_stopping:
        early_stopping_save_dir = os.path.join(
            args.save_dir, "early_stopping")
        if not os.path.exists(early_stopping_save_dir):
            os.makedirs(early_stopping_save_dir)
        early_stopping_save_path = os.path.join(
            early_stopping_save_dir, 'checkpoint.pt')
        if show_bar:
            def trace_func(*args, **kwargs):
                with logging_redirect_tqdm([logger.logger]):
                    logger.info(*args, **kwargs)
        else:
            trace_func = logger.info
        early_stopping = EarlyStopping(
            patience=7, verbose=True, trace_func=trace_func,
            path=early_stopping_save_path)

    loss_list = []
    loss_sum = 0
    loss_num = 0
    global_step = 0
    best_step = 0
    best_f1 = 0
    tic_train = time.time()
    epoch_iterator = range(1, args.num_epochs + 1)
    if show_bar:
        train_postfix_info = {'loss': 'unknown'}
        epoch_iterator = tqdm(
            epoch_iterator, desc='Training', unit='epoch')
    for epoch in epoch_iterator:
        train_data_iterator = train_data_loader
        if show_bar:
            train_data_iterator = tqdm(train_data_iterator,
                                       desc=f'Training Epoch {epoch}', unit='batch')
            train_data_iterator.set_postfix(train_postfix_info)
        for batch in train_data_iterator:
            if show_bar:
                epoch_iterator.refresh()
            input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
            if args.device == 'gpu':
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                att_mask = att_mask.cuda()
                start_ids = start_ids.cuda()
                end_ids = end_ids.cuda()
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=att_mask)
            start_prob, end_prob = outputs[0], outputs[1]
            # start_ids = paddle.cast(start_ids, 'float32')
            # end_ids = paddle.cast(end_ids, 'float32')
            start_ids = start_ids.type(torch.float32)
            end_ids = end_ids.type(torch.float32)
            loss_start = criterion(start_prob, start_ids)
            loss_end = criterion(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss.backward()
            optimizer.step()
            # optimizer.clear_grad()
            optimizer.zero_grad()
            loss_list.append(float(loss))
            loss_sum += float(loss)
            loss_num += 1

            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = loss_sum / loss_num

                if show_bar:
                    train_postfix_info.update({
                        'loss': f'{loss_avg:.5f}'
                    })
                    train_data_iterator.set_postfix(train_postfix_info)
                else:
                    logger.info(
                        "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg,
                           args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                save_dir = os.path.join(
                    args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model_to_save = model
                model_to_save.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

                dev_loss_avg, precision, recall, f1 = evaluate(
                    model, metric, data_loader=dev_data_loader, device=args.device, loss_fn=criterion)

                if show_bar:
                    train_postfix_info.update({
                        'F1': f'{f1:.3f}',
                        'dev loss': f'{dev_loss_avg:.5f}'
                    })
                    train_data_iterator.set_postfix(train_postfix_info)
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f"
                                    % (precision, recall, f1))
                else:
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f"
                                % (precision, recall, f1))
                # Save model which has best F1
                if f1 > best_f1:
                    if show_bar:
                        with logging_redirect_tqdm([logger.logger]):
                            logger.info(
                                f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                            )
                    else:
                        logger.info(
                            f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                        )
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, "model_best")
                    model_to_save = model
                    model_to_save.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                tic_train = time.time()

                if args.early_stopping:
                    # Early Stopping
                    early_stopping(dev_loss_avg, model)
                    if early_stopping.early_stop:
                        if show_bar:
                            with logging_redirect_tqdm([logger.logger]):
                                logger.info("Early stopping")
                        else:
                            logger.info("Early stopping")
                        save_dir = early_stopping_save_path
                        model_to_save = model
                        model_to_save.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        break


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default=None, required=True,
                        type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, required=True,
                        type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. "
                        "Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int,
                        help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=10,
                        type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int,
                        help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--model", default="uie_base_pytorch", type=str,
                        help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--early_stopping", action='store_true', default=False,
                        help="Use early stopping while training")

    args = parser.parse_args()

    do_train()
