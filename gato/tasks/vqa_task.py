# Assume all datasets are downloaded and available from local directories
from gato.tasks.task import Task

import os
from PIL import Image
import io # need to use BytesIO

import numpy as np
import math
import torch
from torch.nn import functional as F
from torch import nn
import json
import random
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GPT2Tokenizer

class VqaTask(Task): 
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader):
        """
        vqa_dataset is the directory where the data for vqa task is located, should end with "/" 
        """
        super().__init__()
        self.train_dataloader = train_dataloader
        self.train_dataloader_iterator = iter(train_dataloader)
        self.test_dataloader = test_dataloader
        self.test_dataloader_iterator = iter(test_dataloader)

    def sample_batch(self):
        return next(self.train_dataloader_iterator)

    def evaluate(self, model, num_examples_to_test=10, deterministic=True, log_examples_to_output=False):
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0

        if log_examples_to_output:
            print(f'--- examples ---')

        last_logged = 0
        num_tested = 0
        while num_tested < num_examples_to_test:
            batch = next(self.test_dataloader_iterator)
            target_tokens = model.text_tokenizer(batch['answer'][0], padding=True, max_length=10)['input_ids']

            # Generate prediction
            pred_logits, pred_answer = model.predict_answer(
                batch['image'][[0]],
                batch['question'][0],
                max_length=len(target_tokens),
                deterministic=deterministic
            )
            if log_examples_to_output and num_tested - last_logged > 2:
                last_logged = num_tested
                print(f'Target answer: {batch["answer"][0]} \n Predicted answer : {pred_answer}')
                print("----")

            # Calculate loss
            loss = loss_fn(pred_logits, torch.tensor(target_tokens).to(model.device))
            total_loss += loss.item()
            total_tokens += len(target_tokens)
            num_tested += 1
        if log_examples_to_output:
            print(f'--- examples end ---')

        avg_loss = total_loss / num_examples_to_test
        perplexity = torch.exp(torch.tensor(avg_loss))

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity.item()
        }
        return metrics


# test code
if __name__ == '__main__':
    # replace the following directories and files names with your directories and files
    task = VqaTask(vqa_dataset              = '/home/<user name>/Git/NEKO/VQA_Data/',
                   train_data               = ['train2014'], 
                   test_data                = ['val2014'],
                   train_img_name_prefix    = ['COCO_train2014_'], 
                   train_img_file_name_len  = [27], 
                   test_img_name_prefix     = ['COCO_val2014_'], 
                   test_img_file_name_len   = [25],
                   questions_file           = 'questions.json', 
                   annotations_file         = 'annotations.json'
                   )

    batch = task.sample_batch(5)
    print(type(batch))
    print(list(batch[0].keys()))
    print(batch[0]['images'][0][1][10])
    print(batch[0]['images'][0][2][15])
    print(batch[0]['images'].shape)
    print(batch[0]['text'])

