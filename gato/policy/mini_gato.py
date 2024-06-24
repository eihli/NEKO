from typing import Callable
import random

import minari
import torch
import torch.nn as nn
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import GPT2TokenizerFast


text_tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
text_tokenizer.pad_token = text_tokenizer.eos_token


CONTEXT_WINDOW = 1024


def tokenize(sample):
    return text_tokenizer(
        sample["text"], truncation=True, padding="max_length", max_length=CONTEXT_WINDOW
    )


####
#### Datasets and Tokenization
####


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


def demo_text_dataloader():
    text_dataset = (
        load_dataset(path="wikitext", name="wikitext-2-v1", streaming=True)
        .filter(not_empty)
        .map(tokenize, batched=True, batch_size=1000)
    )
    text_dataloader = DataLoader(
        text_dataset["train"], batch_size=8, collate_fn=collate_fn
    )
    text_batch = next(iter(text_dataloader))

    print(f"Example of text_batch batch:")
    print(f"Keys: {text_batch.keys()}")
    print(
        f'Shape: text_batch["input_ids"] {text_batch["input_ids"].shape} text_batch["attention_mask"] {text_batch["attention_mask"].shape}'
    )


##
## VQA Datasets, DataLoaders, and Transforms
##
def images_to_patches(images, patch_size=16):
    batch_size, channels, height, width = images.shape
    assert (
        height % patch_size == 0 and width % patch_size == 0
    ), "Image dimensions must be divisible by the patch size"
    # Unfold the height and width dimensions
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Reshape to (batch_size, channels, number_of_patches, patch_size, patch_size)
    patches = patches.contiguous().view(
        batch_size, channels, -1, patch_size, patch_size
    )
    # Rearrange the patches to (batch_size, number_of_patches, patch_size*patch_size*channels)
    patches = patches.permute(0, 2, 3, 4, 1).contiguous()
    patches = patches.view(batch_size, -1, patch_size * patch_size * channels)
    return patches


def patches_to_image(patches, image_shape, patch_size=16):
    batch_size, num_patches, patch_dim = patches.shape
    channels = patch_dim // (patch_size * patch_size)
    height, width = image_shape[-2:]
    assert num_patches == (height // patch_size) * (
        width // patch_size
    ), "Number of patches doesn't match image size"
    assert (
        patch_dim == channels * patch_size * patch_size
    ), "Patch dimensions don't match the expected size"
    # Reshape patches to (batch_size, num_patches, channels, patch_size, patch_size)
    patches = patches.view(batch_size, num_patches, patch_size, patch_size, channels)
    # Permute to (batch_size, channels, num_patches, patch_size, patch_size)
    patches = patches.permute(0, 4, 1, 2, 3).contiguous()
    # Calculate the number of patches along the height and width
    patches_per_row = width // patch_size
    patches_per_col = height // patch_size
    # Reshape to (batch_size, channels, patches_per_col, patches_per_row, patch_size, patch_size)
    patches = patches.view(
        batch_size, channels, patches_per_col, patches_per_row, patch_size, patch_size
    )
    # Permute and reshape to the original image shape
    images = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    images = images.view(batch_size, channels, height, width)
    return images


def normalize_to_between_minus_one_plus_one(t: torch.Tensor):
    min_val = t.min()
    max_val = t.max()
    if min_val == max_val:
        return torch.zeros_like(t)
    normalized = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized


def apply_along_dimension(
    func: Callable[[torch.Tensor], torch.Tensor], dim: int, tensor: torch.Tensor
):
    tensor = tensor.transpose(0, dim)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = torch.stack([func(tensor[:, i]) for i in range(tensor.size(1))], dim=1)
    result = result.reshape(shape).transpose(0, dim)
    return result


transform = transforms.Compose([transforms.RandomResizedCrop((256, 256))])


def vqa_img_transform(sample):
    """These start out as PIL images that are H x W x RGB. Our model expects
    tensors of RGB x H x W. I'm also using 256 x 256 as example H x W (in the
    transforms.Compose transform above), but it could be anything, as long as
    you update the code that creates patches. The Gato paper says they used
    16x16 patches.
    """
    # For no particular reason, this transform is operating on a single sample,
    # while the question/answer tokenization transform below is operating on a batch.
    sample["image"] = transform(torch.permute(sample["image"], (2, 0, 1)))
    return sample


def vqa_img_tokenize(sample: torch.Tensor) -> torch.Tensor:
    """Convert images to patches, normalize each patch, then prepare it for
    embedding by reshaping to CxHxW so that we can send it through the conv
    layers of a ResNet block.
    """
    sample["image"] = images_to_patches(sample["image"])
    # Hardcoding as a reminder to do something smarter
    SQUARE_ROOT_OF_PATCH_SIZE = 4
    sample["image"] = (
        apply_along_dimension(
            normalize_to_between_minus_one_plus_one, 2, sample["image"]
        )
        / SQUARE_ROOT_OF_PATCH_SIZE
    )
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
    # TODO: Possible performance improvement. Mentioned in section 3.2 of the
    # Gato paper. It will introduce a dependency-at-a-distance. Assume a context
    # window of 1024 and images with 256 patches (tokens). That leaves us with
    # 768 tokens to spare. Question and answer will take up some of that, but
    # probably not the full amount. Why not give 128 tokens to the question, 128
    # to the answer, making it 512 total, and then use the remaining 512 to
    # shove in an entirely separate sample?
    #
    # In the meantime, we'll still have a dependency-at-a-distance, hardcoded for now.
    # I'll only use a single sample, and I'll take up the full 1024 sequence length
    answer_tokenized = text_tokenizer(
        answer, truncation=True, padding="max_length", max_length=256
    )
    question_tokenized = text_tokenizer(
        question, truncation=True, padding="max_length", max_length=512
    )
    return {
        "image": batch["image"],
        "question": question,
        "question_input_ids": question_tokenized["input_ids"],
        "question_attention_mask": question_tokenized["attention_mask"],
        "answer": answer,
        "answer_input_ids": answer_tokenized["input_ids"],
        "answer_attention_mask": answer_tokenized["attention_mask"],
    }


def demo_vqa_dataloader():
    vqa_dataset = load_dataset("eihli/micro-ok-vqa", streaming=True).with_format(
        "torch"
    )
    vqa_dataloader = DataLoader(
        vqa_dataset["train"]
        .map(vqa_img_transform)
        .map(vqa_qa_transform, batched=True, batch_size=8)
        .map(vqa_img_tokenize, batched=True, batch_size=8),
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
            [torch.as_tensor(x.observations) for x in batch], batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch], batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch], batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch], batch_first=True
        ),
    }


def demo_control_dataloader():
    control_dataset = minari.load_dataset("door-human-v2", download=True)
    control_dataloader = DataLoader(
        control_dataset, batch_size=8, collate_fn=minari_collate_fn, shuffle=True
    )
    control_batch = next(iter(control_dataloader))

    print(f"Example of Control batch:")
    print(f"Keys: {control_batch.keys()}")
    print(
        f'Shape: control_batch["observations"] {control_batch["observations"].shape} control_batch["rewards"] {control_batch["rewards"].shape}'
    )


####
#### Embedding
####

##
## Text, discrete, and continuous-valued observations and actions
##
EMBEDDING_DIM = 768
_lookup_embedding = nn.Embedding(text_tokenizer.vocab_size, EMBEDDING_DIM)


def lookup_embedding(tokens: torch.Tensor) -> torch.Tensor:
    return _lookup_embedding(tokens)


##
## Image embedding
##


# From section 2.2 of the Gato paper:
#
#    Tokens belonging to image patches for any time-step are embedded using a
#    single ResNet (He et al., 2016a) block to obtain a vector per patch. For
#    image patch token embeddings, we also add a learnable within-image position
#    encoding vector.
class ResNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super(ResNetV2Block, self).__init__()
        self.gn1 = nn.GroupNorm(1, in_channels)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        B, P, HWC = x.shape
        # TODO: Remove these hardcoded values.
        out = x.view(B, P, 16, 16, 3).permute(0, 1, 4, 2, 3)
        B, P, C, H, W = out.shape
        out = self.gn1(out.view(B * P, C, H, W))
        out = self.gelu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.gelu(out)
        out = self.conv2(out)
        return x + out.view(B, P, C, H, W).permute(0, 1, 3, 4, 2).view(B, P, HWC)


_image_embedding = ResNetV2Block(3, EMBEDDING_DIM)


def image_embedding(tokens: torch.Tensor) -> torch.Tensor:
    return _image_embedding(tokens)


####
#### Sequencing
####


## Sequencing is needed because some modalities don't have an inherent order of
## all of their inputs. Text does. It's just the order of the words in the text.
## But what order should the tokens be for a question/image pair? Should the
## image tokens come first? Or the question?

####
#### The preparation process
####
def prepare_text(batch):
    embeddings = lookup_embedding(batch["input_ids"])
    # No special sequencing needs to be done for text.
    return embeddings


def prepare_vqa(batch):
    image_embeddings = image_embedding(batch["image"])
    question_embeddings = lookup_embedding(batch["question_input_ids"])
    answer_embeddings = lookup_embedding(batch["answer_input_ids"])
    sequence = torch.concat([question_embeddings, image_embeddings, answer_embeddings], dim=1)
    return sequence


if __name__ == "__main__":
    demo_text_dataloader()
    demo_vqa_dataloader()
    demo_control_dataloader()
