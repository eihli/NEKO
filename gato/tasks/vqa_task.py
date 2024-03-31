from collections.abc import Callable
from typing import Any
from gato.tasks.task import Task

import torch
from torch import nn
import random
from torch.utils.data import DataLoader
import datasets
from datasets import Dataset

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

class VqaAnswerTransform():
    def __call__(self, batch):
        answers = batch['answers']
        del batch['answers']
        i = random.randint(0, len(answers[0])-1)
        batch['answer'] = [a[i]['answer'] for a in answers]
        return batch

# test code
if __name__ == '__main__':
    from torchvision.transforms import v2 as transforms
    # replace the following directories and files names with your directories and files
    ok_vqa_builder = datasets.load_dataset_builder('HuggingFaceM4/OK-VQA', trust_remote_code=False)
    ok_vqa_builder.download_and_prepare(file_format='arrow')
    ds = ok_vqa_builder.as_dataset()
    vqa_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),  # Would you believe order matters here?
                                            # https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_transform.py#L57
        transforms.PILToTensor(),
        VqaAnswerTransform(),
    ])
    ds.set_transform(vqa_transforms)
    train = ds['train']
    valid = ds['validation']
    train = ds['train'].shuffle(seed=42)
    valid = ds['validation'].shuffle(seed=42)
    train_data_loader = DataLoader(train, batch_size=4, shuffle=False)
    valid_data_loader = DataLoader(valid, batch_size=4, shuffle=False)
    task = VqaTask(train_data_loader, valid_data_loader)
    batch = task.sample_batch()
    print(type(batch))
    print(list(batch.keys()))
    print(f'image shape: {batch["image"].shape}')
    print(batch['question'])
    print(batch['answer'])
