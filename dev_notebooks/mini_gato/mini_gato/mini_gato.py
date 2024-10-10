from typing import Callable
from dataclasses import dataclass, fields
from functools import partial
from itertools import cycle
import os
from pathlib import Path
import pdb
import random
import re
import tempfile
from einops import rearrange
import datasets
import numpy as np
import matplotlib.pyplot as plt
import minari
from minigrid.core import constants as mgc
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model

random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9999"

    # initialize the process group
    init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    destroy_process_group()


# Note on shapes:
# You're probably familiar with the old (B, T, C, ...) shape – batch, timestep, channel.
@dataclass
class TokenData:
    tokens: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor
    embedding: torch.Tensor = torch.tensor([])  # Optional at first.

    def combine(self, other):
        """Concats attributes of self to attributes of other."""
        # Requires padding to already be handled.
        # Requires shapes to be (T', T, [C, ...])
        # Where T' is episode timestep and T is the usual timestep.
        return type(self)(
            tokens=torch.concat([self.tokens, other.tokens]),
            targets=torch.concat([self.targets, other.targets]),
            attention_mask=torch.concat([self.attention_mask, other.attention_mask]),
            embedding=torch.concat([self.embedding, other.embedding]),
        )

    def embed(self, embedder):
        """Populate self.embedding."""
        return type(self)(
            tokens=self.tokens,
            targets=self.targets,
            attention_mask=self.attention_mask,
            embedding=self.embedder_fn(embedder)(self.tokens),
        )

    def to(self, device):
        """Move all attributes to device."""
        return type(self)(
            tokens=self.tokens.to(device),
            targets=self.targets.to(device),
            attention_mask=self.attention_mask.to(device),
            embedding=self.embedding.to(device),
        )

    @property
    def size(self):
        """The number of tokens this will consume of the context window."""
        return self.tokens.size(0) * self.tokens.size(1)


class TextTokenData(TokenData):
    def embedding_fn(self, embedder):
        return embedder.text


class ImageTokenData(TokenData):
    def embedding_fn(self, embedder):
        return embedder.image


class DiscreteTokenData(TokenData):
    def embedding_fn(self, embedder):
        return embedder.discrete


@dataclass
class EpisodeData:
    def __getitem__(self, i):
        # Iterate over fields
        return type(self)(
            **{
                field.name: type(getattr(self, field.name))(
                    tokens=getattr(self, field.name).tokens[[i]],
                    targets=getattr(self, field.name).targets[[i]],
                    attention_mask=getattr(self, field.name).attention_mask[[i]],
                )
                for field in fields(self)
            }
        )

    def combine(self, other):
        return type(self)(
            **{
                field.name: getattr(self, field.name).combine(
                    getattr(other, field.name)
                )
                for field in fields(self)
            }
        )

    @property
    def size(self):
        return sum(getattr(self, field.name).size for field in fields(self))

    @property
    def num_timesteps(self):
        return next(getattr(self, field.name) for field in fields(self)).tokens.size(0)

    def embed(self, embedder):
        return type(self)(
            **{
                field.name: getattr(self, field.name).embed(embedder)
                for field in fields(self)
            }
        )

    def to(self, device):
        return type(self)(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )

    def sequence(self, embeddings):
        raise NotImplementedError()


@dataclass
class FourRoomsTimestep(EpisodeData):
    mission: TextTokenData  # torch.Size((length of episode subsequence, length of _max_ (pad) mission text tokens))
    image: ImageTokenData
    direction: DiscreteTokenData
    actions: DiscreteTokenData

    def sequence(self, sequence_length):
        xs = torch.concat(
            [
                self.mission.embedding,
                self.image.embedding,
                self.direction.embedding,
                self.actions.embedding,
            ],
            dim=1,
        )
        ys = torch.concat(
            [
                self.mission.targets,
                self.image.targets,
                self.direction.targets,
                self.actions.targets,
            ],
            dim=1,
        )
        ms = torch.concat(
            [
                self.mission.attention_mask,
                self.image.attention_mask,
                self.direction.attention_mask,
                self.actions.attention_mask,
            ],
            dim=1,
        )
        T, S, C = xs.shape
        xs, ys, ms = xs.reshape(T * S, C), ys.reshape(T * S), ms.reshape(T * S)
        padding_len = sequence_length - T * S
        xs = F.pad(xs, (0, 0, 0, padding_len), value=0)
        ys, ms = [F.pad(x, (0, padding_len), value=0) for x in [ys, ms]]
        return xs, ys, ms


@dataclass
class TextTimestep(EpisodeData):
    text: TextTokenData

    def sequence(self, _):
        return (
            self.text.embedding.squeeze(0),
            self.text.targets.squeeze(0),
            self.text.attention_mask.squeeze(0),
        )


@dataclass
class VQATimestep(EpisodeData):
    question: TextTokenData
    image: ImageTokenData
    answer: TextTokenData

    def sequence(self, sequence_length):
        xs = torch.concat(
            [self.question.embedding, self.image.embedding, self.answer.embedding],
            dim=1,
        )
        ys = torch.concat(
            [self.question.targets, self.image.targets, self.answer.targets], dim=1
        )
        ms = torch.concat(
            [
                self.question.attention_mask,
                self.image.attention_mask,
                self.answer.attention_mask,
            ],
            dim=1,
        )
        T, S, C = xs.shape
        xs, ys, ms = xs.reshape(T * S, C), ys.reshape(T * S), ms.reshape(T * S)
        padding_len = sequence_length - T * S
        xs = F.pad(xs, (0, 0, 0, padding_len), value=0)
        ys, ms = [F.pad(x, (0, padding_len), value=0) for x in [ys, ms]]
        return xs, ys, ms


class Tokenizer:
    def __init__(self, text_gen_tokenizer, text_obs_tokenizer):
        self.text_gen_tokenizer = text_gen_tokenizer
        self.text_obs_tokenizer = text_obs_tokenizer

    @property
    def bos_token(self):
        return self.text_gen_tokenizer.func.bos_token

    @property
    def eos_token(self):
        return self.text_gen_tokenizer.func.eos_token

    def text_gen(self, data, **kwargs):
        """Tokenize text, return ones attention mask."""
        tokenized = self.text_gen_tokenizer(data, **kwargs)
        return TextTokenData(
            **{
                "tokens": tokenized["input_ids"][:, :-1].to(torch.long),
                "targets": tokenized["input_ids"][:, 1:].to(torch.long),
                "attention_mask": tokenized["attention_mask"][:, :-1],
            }
        )

    def text_obs(self, data, **kwargs):
        """Tokenize text, return zeros attention mask."""
        tokenized = self.text_obs_tokenizer(data, **kwargs)
        return TextTokenData(
            **{
                "tokens": tokenized["input_ids"].to(torch.long),
                "targets": tokenized["input_ids"].to(torch.long),
                "attention_mask": torch.zeros_like(tokenized["attention_mask"]),
            }
        )

    def image(self, data):
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        patches = images_to_patches(data, patch_size=16)
        # Hardcoding math.sqrt(16). TODO: Handle other patch sizes.
        SQUARE_ROOT_OF_PATCH_SIZE = 3.464
        xs = (
            apply_along_dimension(normalize_to_between_minus_one_plus_one, 2, patches)
            / SQUARE_ROOT_OF_PATCH_SIZE
        )
        # We don't predict images, but we need ys
        # becaues these image ys will be in our
        # concatenated ys of text/image/action/etc...
        ys = torch.zeros(xs.shape[:2]).to(torch.long)
        ms = torch.zeros(xs.shape[:2])  # Same story as above.
        return ImageTokenData(tokens=xs, targets=ys, attention_mask=ms)

    def discrete_obs(self, data):
        if len(data.shape) == 0:
            data = data.unsqueeze(0)
        if len(data.shape) == 1:
            data = data.unsqueeze(1)
        xs = data
        ys = torch.zeros(xs.shape[:2])
        ms = torch.zeros(xs.shape[:2])
        return DiscreteTokenData(tokens=xs, targets=ys, attention_mask=ms)

    def discrete_act(self, data):
        if len(data.shape) == 0:
            data = data.unsqueeze(0)
        if len(data.shape) == 1:
            data = data.unsqueeze(1)
        xs = torch.concat(
            [
                torch.full((data.size(0), 1), 1023),  # Hardcoding 1023 here as the "beginning of action" token.
                data,
            ],
            dim=1,
        )  # Instead of '|' being the separator, like Gato...
        ys = torch.concat(
            [
                data,
                torch.full((data.size(0), 1), 1023),  # Hardcoding 1023 here as the "end of action" token.
            ],
            dim=1,
        )
        ms = torch.ones(*ys.shape)
        return DiscreteTokenData(tokens=xs, targets=ys, attention_mask=ms)

    def continuous(self, data):
        raise NotImplementedError("TODO")


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


# From section 2.2 of the Gato paper:
#
#    Tokens belonging to image patches for any time-step are embedded using a
#    single ResNet (He et al., 2016a) block to obtain a vector per patch. For
#    image patch token embeddings, we also add a learnable within-image position
#    encoding vector.
class ResNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=24):
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
        B, T, CHW = x.shape
        # TODO: Remove these hardcoded values.
        out = rearrange(x, "b t (c h w) -> (b t) c h w", c=3, h=16)
        out = self.gn1(out)
        out = self.gelu(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.gelu(out)
        out = self.conv2(out)
        return x + rearrange(out, "(b t) c h w -> b t (c h w)", b=B, t=T)


def images_to_patches(images, patch_size=16):
    return rearrange(
        images, "b c (h s1) (w s2) -> b (h w) (c s1 s2)", s1=patch_size, s2=patch_size
    )


def normalize_to_between_minus_one_plus_one(t: torch.Tensor):
    min_val, max_val = t.min(), t.max()
    if min_val == max_val:
        return torch.zeros_like(t)
    normalized = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized


# There's a small deviation in the NEKO codebase from the paper.
# The paper normalizes _per patch_. The NEKO codebase currently normalizes _per image_.
# https://github.com/eihli/NEKO/blob/master/gato/policy/embeddings.py#L38
# This notebook normalizeds per patch. That's what this utility helps.
def apply_along_dimension(func, dim, tensor):
    tensor = tensor.transpose(0, dim)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = torch.stack([func(tensor[:, i]) for i in range(tensor.size(1))], dim=1)
    result = result.reshape(shape).transpose(0, dim)
    return result


# Create lookup table for turning minigrid maps into images.
lut = np.zeros((256, 3), dtype=np.uint8)
for idx, color_name in mgc.IDX_TO_COLOR.items():
    lut[idx] = mgc.COLORS[color_name]


def minigrid_to_rgb(episode):
    """Convert discrete "image" observations into actual images.
    I'm expecting this will improve our image modality while not losing
    much. The downside is we can fit less in our context window. Note:
    We might need to overlay the color/type image (index 1) with the
    state image (index 2), if we really don't want to lose any info."""
    # Apply lookup to second channel
    image = lut[episode.observations["image"][:, :, :, 1]]
    # Convert to PyTorch tensor and permute
    image = torch.from_numpy(image).permute(0, 3, 1, 2)
    return image


image_transform = transforms.Compose(
    [
        # No particular reason to use `transforms.Compose` here since we're only doing one transform. But it's nice to know about.
        transforms.RandomResizedCrop((192, 192), (0.5, 1.0)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def minigrid_tokenizer(tokenizer, episode):
    num_timesteps = len(episode.actions)
    image = image_transform(minigrid_to_rgb(episode)[:num_timesteps])
    image = tokenizer.image(image[:num_timesteps])
    mission = tokenizer.text_obs(
        episode.observations["mission"][:num_timesteps], padding=False
    )
    direction = tokenizer.discrete_obs(
        torch.from_numpy(episode.observations["direction"])[:num_timesteps]
    )
    actions = tokenizer.discrete_act(torch.from_numpy(episode.actions))
    return FourRoomsTimestep(
        mission=mission, image=image, direction=direction, actions=actions
    )


def vqa_tokenizer(tokenizer, sample):
    image = tokenizer.image(image_transform(sample["image"]))
    # Pad to longest. Not max length. Max length defaults to sequence length. We need space for the question and answer.
    answer = tokenizer.text_gen(
        random.choice(sample["answers"])["answer"], padding="longest"
    )
    question = tokenizer.text_obs(sample["question"])
    return VQATimestep(question=question, image=image, answer=answer)


def minigrid_collate_fn(sequence_length, batch):
    result = []
    for sample in batch:
        i = random.randint(0, sample.num_timesteps - 1)
        # Starting at that index, we'll continue adding observations to our context window until
        # we run out of space.
        step = sample[i]
        i += 1
        while (
            i < len(sample.actions.tokens)
            and step.size + step[0].size < sequence_length
        ):
            step = step.combine(sample[i])
            i += 1
        result.append(step)
    return result


def text_tokenizer(tokenizer, text):
    return TextTimestep(
        text=tokenizer.text_gen(tokenizer.bos_token + text + tokenizer.eos_token)
    )


@dataclass
class Embedder:
    text: Callable
    image: Callable
    discrete: Callable


@dataclass
class MiniGatoConfig:
    embedding_dim: int
    sequence_length: int
    vocab_size: int
    tokenizer: Tokenizer
    transformer_config: GPT2Config
    transformer: GPT2Model


class MiniGato(nn.Module):
    def __init__(self, config: MiniGatoConfig):
        super().__init__()
        self.config = config
        self.sequence_length = self.config.sequence_length
        text_embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        image_embedding = ResNetV2Block(3, self.config.embedding_dim)
        discrete_embedding = nn.Embedding(1024, self.config.embedding_dim)
        self.embedder = Embedder(
            text=text_embedding, image=image_embedding, discrete=discrete_embedding
        )
        self.transformer = self.config.transformer
        self.lm_head = nn.Linear(
            self.transformer.config.hidden_size, self.config.vocab_size
        )

    def forward(self, batch):
        batch = [
            sample.embed(self.embedder).sequence(self.sequence_length)
            for sample in batch
        ]
        xs, ys, ms = map(torch.stack, zip(*batch))
        xs, ys, ms = [x.to(device) for x in [xs, ys, ms]]
        out = self.transformer(inputs_embeds=xs)
        predicted = self.lm_head(out.last_hidden_state)
        return predicted, ys, ms


def infinite_dataloader(fn):
    it = iter(fn())
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(fn())


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
    target = target.view(-1).to(torch.long)
    losses = F.cross_entropy(predicted, target, reduction="none")
    losses = losses * mask.squeeze(-1).view(-1)
    loss = losses.sum() / (mask.sum() + 1e-8)
    return loss


class MiniGatoTrainer:
    def __init__(
        self, model, optimizer, dataloaders, scheduler=None, lr=3e-4, num_iterations=10
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scheduler = scheduler
        self.dl_it = cycle(dataloaders)
        self.losses = []
        self.num_iterations = num_iterations
        self.lr = lr

    def train(self):
        self.model.train()
        for i in tqdm(range(self.num_iterations)):
            dl = next(self.dl_it)
            batch = next(dl)
            self.optimizer.zero_grad()
            predicted, targets, attention_mask = self.model(batch)
            loss = cross_entropy(predicted, targets, attention_mask)
            self.losses.append(loss.item())
            loss.backward()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.step()


def acquire_shakespeare_dataset():
    temp_dir = tempfile.gettempdir()
    shakespeare_filepath = Path(temp_dir) / "shakespeare.txt"
    if not os.path.exists(shakespeare_filepath):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(shakespeare_filepath, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

    with open(shakespeare_filepath, "r", encoding="utf-8") as f:
        data = f.read()

    # Split the dataset into each character's lines.
    # Continue taking lines until you have at least 250 words in the sample.
    # Add that sample to the dataset.
    characters_lines = re.split(r"\n\s*\n", data.strip())
    MIN_WORDS_PER_BATCH = 250
    sample = [characters_lines[0]]
    num_words_in_sample = len(characters_lines[0].split())
    text_dataset = []
    i = 1
    while i < len(characters_lines):
        if num_words_in_sample > MIN_WORDS_PER_BATCH:
            text_dataset.append("\n\n".join(sample))
            num_words_in_sample -= len(sample[0].split())
            sample = sample[1:]
        sample += [characters_lines[i]]
        num_words_in_sample += len(characters_lines[i].split())
        i += 1

    return text_dataset


def text_tokenizer(tokenizer, text):
    return TextTimestep(
        text=tokenizer.text_gen(tokenizer.bos_token + text + tokenizer.eos_token)
    )


def init_default_config() -> MiniGatoConfig:
    SEQUENCE_LENGTH = 1024
    __text_tokenizer = GPT2Tokenizer.from_pretrained(
        "openai-community/gpt2", clean_up_tokenization_spaces=True
    )
    __text_tokenizer.pad_token = __text_tokenizer.eos_token
    _text_gen_tokenizer = partial(
        __text_tokenizer,
        max_length=SEQUENCE_LENGTH + 1,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    _text_obs_tokenizer = partial(
        __text_tokenizer,
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    transformer_config = GPT2Config()
    tokenizer = Tokenizer(_text_gen_tokenizer, _text_obs_tokenizer)
    return MiniGatoConfig(
        embedding_dim=768,
        sequence_length=1024,
        vocab_size=__text_tokenizer.vocab_size,
        transformer_config=transformer_config,
        transformer=GPT2Model(transformer_config),
        tokenizer=tokenizer,
    )


default_config = init_default_config()


def main():
    NUM_ITERATIONS = 10
    BATCH_SIZE = 6
    LR = 1e-3

    config = init_default_config()
    tokenizer = config.tokenizer
    minigrid_tokenize = partial(minigrid_tokenizer, tokenizer)
    vqa_tokenize = partial(vqa_tokenizer, tokenizer)

    shakespeare_dataset = acquire_shakespeare_dataset()
    vqa_dataset = datasets.load_dataset("eihli/micro-ok-vqa").with_format("pt")
    minigrid_dataset = minari.load_dataset("D4RL/minigrid/fourrooms-v0", download=True)

    text_tokenize = partial(text_tokenizer, tokenizer)
    shakespeare_dataset_xf = TransformDataset(shakespeare_dataset, text_tokenize)
    minigrid_dataset_xf = TransformDataset(minigrid_dataset, minigrid_tokenize)
    vqa_dataset_xf = TransformDataset(vqa_dataset["train"], vqa_tokenize)

    dataloaders = [
        infinite_dataloader(
            partial(
                DataLoader,
                minigrid_dataset_xf,
                batch_size=BATCH_SIZE,
                collate_fn=partial(minigrid_collate_fn, config.sequence_length),
            )
        ),
        infinite_dataloader(
            partial(
                DataLoader,
                shakespeare_dataset_xf,
                batch_size=BATCH_SIZE,
                collate_fn=lambda x: x,
            )
        ),
        infinite_dataloader(
            partial(
                DataLoader,
                vqa_dataset_xf,
                batch_size=BATCH_SIZE,
                collate_fn=lambda x: x,
            )
        ),
    ]

    config = init_default_config()
    model = MiniGato(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_ITERATIONS, eta_min=1e-6
    )
    trainer = MiniGatoTrainer(
        model,
        optimizer,
        dataloaders,
        num_iterations=NUM_ITERATIONS,
        lr=1e-3,
    )

    trainer.train()


if __name__ == "__main__":
    main()
