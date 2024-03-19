import lightning as L
import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader

class AGNewsDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        tokenizer = get_tokenizer("basic_english")
        train_iter = AG_NEWS(split="train")
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        self.vocab_size = len(vocab)
        self.num_class = len(set([label for (label, text) in train_iter])) # 4 in AG_NEWS
        self.text_pipeline = lambda x: vocab(tokenizer(x))
        self.label_pipeline = lambda x: int(x) - 1

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list, text_list, offsets

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_iter = AG_NEWS(split="train")
            train_dataset = to_map_style_dataset(train_iter)
            num_train = int(len(train_dataset) * 0.95)
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [num_train, len(train_dataset) - num_train], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            test_iter = AG_NEWS(split="test")
            self.test_dataset = to_map_style_dataset(test_iter)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_batch)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_batch)