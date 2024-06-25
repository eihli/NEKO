from typing import Callable, Tuple
import random

import minari
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import GPT2TokenizerFast, GPT2Config, GPT2Model


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
    input_ids = torch.tensor([s["input_ids"] for s in batch])
    attention_mask = torch.tensor([s["attention_mask"] for s in batch])
    return {
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
        "question_input_ids": question_tokenized["input_ids"],
        "question_attention_mask": question_tokenized["attention_mask"],
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
        .map(vqa_img_tokenize, batched=True, batch_size=8, remove_columns=["answers", "question", "answer_type", "question_type", "confidence"]),
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
        return x + out.reshape(B, P, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, P, HWC)


_image_embedding = ResNetV2Block(3, EMBEDDING_DIM)


def image_embedding(tokens: torch.Tensor) -> torch.Tensor:
    return _image_embedding(tokens)


####
#### Embedding and sequencing
####

## Sequencing is needed because some modalities don't have an inherent order of
## all of their inputs. Text does. It's just the order of the words in the text.
## But what order should the tokens be for a question/image pair? Should the
## image tokens come first? Or the question?


## Targets
##
## Since we're doing text prediction, our targets are just going to be the
## offset-by-one of the inputs. We could have included those in the data loader,
## or we can calculate them later.
def targets(t: torch.Tensor) -> torch.Tensor:
    tail = torch.full((t.size(0), 1), text_tokenizer.eos_token_id, dtype=torch.long, device=t.device)
    targets = torch.concat([t.type(torch.long)[:, 1:], tail], dim=1)
    return targets


def embed_and_sequence_text(
    batch: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = lookup_embedding(batch["input_ids"])
    # No special sequencing needs to be done for text.
    return (
        embeddings,
        batch["attention_mask"].unsqueeze(-1),
        targets(batch["input_ids"]),
    )


def embed_and_sequence_vqa(
    batch: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_embeddings = image_embedding(batch["image"])
    question_embeddings = lookup_embedding(batch["question_input_ids"])
    question_targets = targets(batch["question_input_ids"])
    answer_embeddings = lookup_embedding(batch["answer_input_ids"])
    answer_targets = targets(batch["answer_input_ids"])
    embeddings = torch.concat(
        [question_embeddings, image_embeddings, answer_embeddings], dim=1
    )
    target_sequence = torch.concat(
        [
            question_targets,
            torch.zeros(image_embeddings.shape[:2], dtype=torch.long, device=question_targets.device),
            answer_targets,
        ],
        dim=1,
    )
    attention_mask = torch.concat(
        [
            batch["question_attention_mask"],
            torch.zeros(batch["image"].shape[:2], device=batch["question_attention_mask"].device),
            batch["answer_attention_mask"],
        ],
        dim=1,
    )
    return embeddings, attention_mask.unsqueeze(-1), target_sequence


## Loss
##
## See section 2.3 of the Gato paper.
##
##   Let b index a training batch of sequences B. We define a masking function m
##   such that m(b, l) = 1 if the token at index l is either from text or from
##   the logged action of an agent, and 0 otherwise. The training loss for a
##   batch B can then be written as...
def cross_entropy(predicted, target, mask):
    # See: https://youtu.be/kCc8FmEb1nY?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=1553
    B, T, C = predicted.shape
    predicted = predicted.view(B * T, C)
    target = target.view(-1)
    losses = F.cross_entropy(predicted, target, reduction="none")
    losses = losses * mask.squeeze(-1).view(-1)
    loss = losses.sum() / mask.sum()
    return loss


####
#### The transformer
####
def init_model():
    configuration = GPT2Config(
        n_layer=6,
        n_head=6,
        n_embd=768
    )
    model = GPT2Model(configuration)
    return model


def init_optimizer(params):
    optimizer = torch.optim.AdamW(params)
    return optimizer


####
#### Training
####
## We need to remove the embedding layer from the GPT2 model because we're
## sending it other than just text.
class IdentityEmbedding(torch.nn.Module):
    def forward(self, input):
        return input


def remove_embedding_layer_from_model(model):
    model.set_input_embeddings(IdentityEmbedding())


def train(model):
    global _lookup_embedding, _image_embedding
    accelerator = Accelerator()

    lm_head = nn.Linear(model.config.hidden_size, text_tokenizer.vocab_size)
    params = (
        list(model.parameters())
        + list(_lookup_embedding.parameters())
        + list(_image_embedding.parameters())
        + list(lm_head.parameters())
    )
    optimizer = init_optimizer(params)

    text_dataset = (
        load_dataset(path="wikitext", name="wikitext-2-v1", streaming=True)
        .filter(not_empty)
        .map(tokenize, batched=True, batch_size=1000)
    )
    text_dataloader = DataLoader(
        text_dataset["train"], batch_size=2, collate_fn=collate_fn
    )

    vqa_dataset = load_dataset("eihli/micro-ok-vqa", streaming=True).with_format(
        "torch"
    )
    vqa_dataloader = DataLoader(
        vqa_dataset["train"]
        .map(vqa_img_transform)
        .map(vqa_qa_transform, batched=True, batch_size=8)
        .map(vqa_img_tokenize, batched=True, batch_size=8, remove_columns=["answers", "question", "answer_type", "question_type", "confidence"]),
        batch_size=2,
    )

    model, _lookup_embedding, _image_embedding, lm_head, optimizer, text_dataloader, vqa_dataloader = accelerator.prepare(model, _lookup_embedding, _image_embedding, lm_head, optimizer, text_dataloader, vqa_dataloader)

    text_dataloader = iter(text_dataloader)
    vqa_dataloader = iter(vqa_dataloader)
    for epoch in range(20):
        text_batch = next(text_dataloader)
        vqa_batch = next(vqa_dataloader)
        text_sequence, text_attention_mask, text_targets = embed_and_sequence_text(text_batch)
        vqa_sequence, vqa_attention_mask, vqa_targets = embed_and_sequence_vqa(vqa_batch)
        x = torch.concat([text_sequence, vqa_sequence])
        y = torch.concat([text_targets, vqa_targets])
        m = torch.concat([text_attention_mask, vqa_attention_mask])
        optimizer.zero_grad()
        o = model(inputs_embeds=x)
        p = lm_head(o.last_hidden_state)
        loss = cross_entropy(p, y, m)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/100], Loss: {loss.item()}")
    return model


if __name__ == "__main__":
    train()
