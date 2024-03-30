from typing import Any
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
import datasets

class VqaTask(Task): 
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader):
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

class IdentityTransform():
    def __call__(self, sample: Any):
        return sample

class VqaDataset(datasets.IterableDataset):
    def __init__(self, dataset: datasets.Dataset, transform=IdentityTransform()):
        self.transform = transform
        self.dataset = dataset.to_iterable_dataset(num_shards=8).shuffle(seed=42)

    def __iter__(self):
        for sample in iter(self.dataset):
            answers = sample['answers']
            del sample['answers']
            sample.update(random.choice(answers))
            yield self.transform(sample)

# test code
if __name__ == '__main__':
    from torchvision.transforms import v2 as transforms
    # replace the following directories and files names with your directories and files
    ok_vqa_builder = datasets.load_dataset_builder('HuggingFaceM4/OK-VQA', trust_remote_code=False)
    ok_vqa_builder.download_and_prepare(file_format='arrow')
    ds = ok_vqa_builder.as_dataset()
    vqa_transforms = transforms.Compose([
        transforms.PILToTensor(),
        transforms.CenterCrop(1024),
        transforms.Resize(size=(256, 256))
    ])
    task = VqaTask(
        DataLoader(VqaDataset(ds['train'], vqa_transforms), batch_size=4),
        DataLoader(VqaDataset(ds['validation'], vqa_transforms), batch_size=4),
    )

    batch = task.sample_batch()
    print(type(batch))
    print(list(batch.keys()))
    print(f'image shape: {batch["image"].shape}')
    print(batch['question'])
    print(batch['answer'])
