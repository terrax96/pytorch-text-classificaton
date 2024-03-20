import lightning as L
from torch import nn
from torch import optim
from torchmetrics import Accuracy


class BaseTextClassificationModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super(BaseTextClassificationModel, self).__init__()
        # this EmbeddingBag layer defines the text embedding as the mean of the embedding of each word in the input text
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class TextClassificationModel(L.LightningModule):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int, lr: float, with_scheduler: bool) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = BaseTextClassificationModel(vocab_size, embed_dim, num_class)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.with_scheduler = with_scheduler
        self.metric = Accuracy(task="multiclass", num_classes=num_class)

    def training_step(self, batch, batch_idx):
        label, text, offsets = batch
        predicted_label = self.model(text, offsets)
        loss = self.criterion(predicted_label, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        label, text, offsets = batch
        predicted_label = self.model(text, offsets)
        acc = self.metric(predicted_label, label)
        self.log('accuracy', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if self.with_scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
                }
            }
        else:
            return optimizer
        

