#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: May - 2021
Project : NLP Research
_____________________________________________________________________________



The Module Has Been Build For Traning GPT2 model
_____________________________________________________________________________
"""

import os
import glob
import time
import random
import argparse
import warnings

from pathlib import Path
from transformers import TextDataset
from transformers import GPT2LMHeadModel
from transformers import Trainer,AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments,AutoModelWithLMHead
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

class BPE_token(object):
    def __init__(self, size = 60000):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()


    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=size, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        self.tokenizer.train(paths, trainer)


    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)    

def get_parse():
    parser = argparse.ArgumentParser(description='GPT-2 base model training.')
    parser.add_argument('--data_path',
                        type = str, 
                        required = True,
                        help = "str - Data directory path")
    
    # Tokenizer args
    parser.add_argument('--new_token',
                        type = bool, 
                        default = True, 
                        help = "bool - Create new tokenizer or not (default: True).")
    parser.add_argument('--vocab_size',
                        type = int, 
                        default = 60000, 
                        help = "int - Size of vocab (Optional if new_token is True, default: 60000).")
    parser.add_argument('--tokenizer_path',
                        type = str, 
                        required = True,
                        help = "str - Tokenizer directory path - \
                        which to save new tokenizer if new_token is True or load trained tokenizer if new_token is False'")
    
    # Training config args
    parser.add_argument('--outdir',
                        type = str,
                        default = '', 
                        help = "str - The output directory - where to save model checkpoint, default: ''")
    parser.add_argument('--train_batch_size',
                        type = int,
                        default = 32, 
                        help = "int - The training batchsize for each device, default: 32")
    parser.add_argument('--eval_batch_size',
                        type = int,
                        default = 32, 
                        help = "int - The evaluation batchsize for each device, default: 32")
    parser.add_argument('--eval_steps',
                        type = int,
                        default = 500, 
                        help = "int - Number of update steps between two evaluations, default: 500")
    parser.add_argument('--save_steps',
                        type = int,
                        default = 1000, 
                        help = "int - After # steps model is saved, default: 1000")
    
    # Dataset config args
    parser.add_argument('--test_path',
                        type = str,
                        required = True,
                        help = "str - The path to testing document, required.")
    parser.add_argument('--block_size',
                        type = int,
                        default = 200,
                        help = "int - The blocksize to training, default: 200")
    
    # Training args
    parser.add_argument('--is_continue',
                        type = str,
                        default = False,
                        help = "bool - Continue from last checkpoint in save path or not, default: False.")
    parser.add_argument('--block_size',
                        type = int,
                        default = 200,
                        help = "int - The blocksize to training, default: 200")

    args = parser.parse_args()

if __name__ == '__main__':
    args = get_parse()

    # Loading tokeinzer model
    data_paths = [str(x) for x in Path(args.data_path).glob("*.txt")]
    if not args.new_token:
        save_path = args.tokenizer_path
        tokenizer = GPT2Tokenizer.from_pretrained(save_path)
        tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
        })
    else:
        vocab_size = args.vocab_size
        save_path = args.tokenizer_path
        tokenizer = BPE_token(vocab_size)
        tokenizer.bpe_train(data_paths)
        tokenizer.save_tokenizer(save_path) 

    # Trainer Configoutdir
    training_args = TrainingArguments(
        output_dir=args.outdir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=arg.seval_batch_size,
        eval_steps = args.eval_steps,
        save_steps=args.save_steps,
        warmup_steps=500,
        prediction_loss_only=True,)

    # Test dataset loader
    data_collator = DataCollatorForLanguageModeling( 
        tokenizer=tokenizer, mlm=False,)
    print("Loading test dataset", end='.............')
    begintime = time.time()
    test_dataset = TextDataset(
      tokenizer=tokenizer,
      file_path=args.test_path,
      block_size=args.block_size,
      overwrite_cache = True)      
    print("Done! Execution time: %.4f"%(time.time() - begintime))

    # Model config
    config = GPT2Config.from_pretrained('gpt2')
    config.vocab_size=tokenizer.vocab_size
    config.bos_token_id=tokenizer.bos_token_id
    config.eos_token_id=tokenizer.eos_token_id
    config.unk_token = tokenizer.unk_token
    config.pad_token = tokenizer.pad_token
    config.mask_token = tokenizer.mask_token
    model = GPT2LMHeadModel(config)

    # Load last checkpoint
    if args.is_continue:
        if os.path.exists(args.outdir):
            saved_dir = sorted([str(x) for x in Path(args.outdir).glob("checkpoint_epochs_*_batch_*.txt")])
            last_save_dir = saved_dir[-1] if saved_dir else None
            start_e = last_save_dir.split("_")[-3] if saved_dir else 0
            start_b = last_save_dir.split("_")[-1] if saved_dir else 0
    else:
        start_e = 0
        start_b = 0

    # Training model
    for e in range(10):
        if e<int(start_e):
            continue
        else:
            start_e = -1
        for ind, original_train_path in enumerate(sorted(args.data_path)):
            if original_train_path == args.text_path:
                continue
            if i<=int(start_b):
                continue
            else:
                start_b = -1
            if last_save_dir is not None:
                model = GPT2LMHeadModel.from_pretrained(last_save_dir)
            print("Loading training dataset for epoch %s, batch %s"%(e,i), end='.............')
            begintime = time.time()
            train_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=original_train_path,
                block_size=200, overwrite_cache = True)
            print("Done! Execution time: %.4f"%(time.time() - begintime))
            trainer = Trainer(
                model=model,
                args=training_args,
            )
            trainer.data_collator=data_collator
            trainer.train_dataset=train_dataset
            trainer.eval_dataset=test_dataset
            trainer.train()
            model_name = 'checkpoint_epochs_%s_batch_%s'%(str(e).zfill(2),str(i).zfill(2))
            last_save_dir = os.path.join(args.outdir,model_name)
            trainer.save_model(output_dir = last_save_dir)    
            