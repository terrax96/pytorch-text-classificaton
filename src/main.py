from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from model import TextClassificationModel
from agnews_data_module import AGNewsDataModule

if __name__ == "__main__":
    ag_news_datamodule = AGNewsDataModule(batch_size=64)
    ag_news_datamodule.prepare_data()
    model = TextClassificationModel(vocab_size=ag_news_datamodule.vocab_size, embed_dim=64, num_class=ag_news_datamodule.num_class, lr=5, with_scheduler=False)

    checkpoint_callback = ModelCheckpoint(monitor="accuracy", mode="max")
    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback], accelerator="gpu", gradient_clip_algorithm="norm", gradient_clip_val=0.1)
    trainer.fit(model=model, datamodule=ag_news_datamodule)
    trainer.test(model=model, datamodule=ag_news_datamodule, ckpt_path=checkpoint_callback.best_model_path)