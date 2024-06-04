import random

import minari
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import GPT2TokenizerFast

transform = transforms.Compose([transforms.RandomResizedCrop((224, 224))])

text_tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
text_tokenizer.pad_token = text_tokenizer.eos_token


def tokenize(sample):
    return text_tokenizer(sample["text"], padding="longest", truncation=True)


##
## Text Datasets, DataLoaders, and Transforms
##
def not_empty(sample):
    return sample["text"] != ""


def collate_fn(batch):
    """Convert batches of tokens to tensors. For details on collate_fn, see:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate I
    need to take a close look at why exactly I needed this. I just remember
    getting some wierd dimensionaltiy issues without it. Like... I was getting
    333 items in a batch, where each item was of length batch_size.

    """
    text = [s["text"] for s in batch]
    input_ids = torch.tensor([s["input_ids"] for s in batch])
    attention_mask = torch.tensor([s["attention_mask"] for s in batch])
    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


text_dataset = (
    load_dataset(path="wikitext", name="wikitext-2-v1", streaming=True)
    .filter(not_empty)
    .map(tokenize, batched=True, batch_size=1000)
)
text_dataloader = DataLoader(text_dataset["train"], batch_size=8, collate_fn=collate_fn)
text_batch = next(iter(text_dataloader))

print(f"Example of text_batch batch:")
print(f"Keys: {text_batch.keys()}")
print(
    f'Shape: text_batch["input_ids"] {text_batch["input_ids"].shape} text_batch["attention_mask"] {text_batch["attention_mask"].shape}'
)


##
## VQA Datasets, DataLoaders, and Transforms
##
def vqa_img_transform(sample):
    """These start out as PIL images that are H x W x RGB. Our model expects
    tensors of RGB x H x W. I'm also using 224 x 224 as example H x W (in the
    transforms.Compose transform above), but it could be anything.
    """
    # For no particular reason, this transform is operating on a single sample,
    # while the question/answer tokenization transform below is operating on a batch.
    sample["image"] = transform(torch.permute(sample["image"], (2, 0, 1)))
    return sample


def vqa_qa_transform(batch):
    """The OK-VQA dataset includes 10 possible answers for every question/image
    pair. This transform randomly selects one of the answers and tokenizes the
    text. We might eventually be able to squeek out a small improvement by
    intelligently choosing an answer - I think they include a "confidence"
    rating for each answer, and we could take a weighted random choice based on
    confidence. But whatever. This is a solid simple start.
    """
    answers = [random.choice(a) for a in batch["answers"]]
    answer = [a["answer"] for a in answers]
    question = batch["question"]
    answer_tokenized = text_tokenizer(answer, padding="longest", truncation=True)
    question_tokenized = text_tokenizer(question, padding="longest", truncation=True)
    return {
        "image": batch["image"],
        "question": question,
        "question_input_ids": question_tokenized["input_ids"],
        "question_attention_mask": question_tokenized["attention_mask"],
        "answer": answer,
        "answer_input_ids": answer_tokenized["input_ids"],
        "answer_attention_mask": answer_tokenized["attention_mask"],
    }

vqa_dataset = load_dataset("eihli/micro-ok-vqa", streaming=True).with_format("torch")
vqa_dataloader = DataLoader(
    vqa_dataset["train"]
    .map(vqa_img_transform)
    .map(vqa_qa_transform, batched=True, batch_size=8),
    batch_size=8,
)
vqa_batch = next(iter(vqa_dataloader))


print(f"Example of VQA batch:")
print(f"Keys: {vqa_batch.keys()}")
print(
    f'Shape: vqa_batch["image"] {vqa_batch["image"].shape} vqa_batch["answer_attention_mask"] {vqa_batch["answer_attention_mask"].shape}'
)


##
## Control Datasets, DataLoaders, and Transforms
##
def minari_collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "seed": torch.Tensor([x.seed for x in batch]),
        "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }

control_dataset = minari.load_dataset("door-human-v2", download=True)
control_dataloader = DataLoader(control_dataset, batch_size=8, collate_fn=minari_collate_fn, shuffle=True)
control_batch = next(iter(control_dataloader))

print(f"Example of Control batch:")
print(f"Keys: {control_batch.keys()}")
print(
    f'Shape: control_batch["observations"] {control_batch["observations"].shape} control_batch["rewards"] {control_batch["rewards"].shape}'
)
