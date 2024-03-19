import torch
from model import TextClassificationModel
from agnews_data_module import AGNewsDataModule

if __name__ == "__main__":
    # prediction example
    ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
    ag_news_datamodule = AGNewsDataModule(batch_size=64)
    ag_news_datamodule.prepare_data()
    model = TextClassificationModel.load_from_checkpoint("lightning_logs/version_1/checkpoints/epoch=8-step=16038.ckpt").model
    model.eval()

    def predict(text, text_pipeline):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1


    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing. \
        Thursday’s first round at the WGC-FedEx St. Jude Invitational \
        was another story. With temperatures in the mid-80s and hardly any \
        wind, the Spaniard was 13 strokes better in a flawless round. \
        Thanks to his best putting performance on the PGA Tour, Rahm \
        finished with an 8-under 62 for a three-stroke lead, which \
        was even more impressive considering he’d never played the \
        front nine at TPC Southwind."

    model = model.to("cpu")

    print("This is a %s news" % ag_news_label[predict(ex_text_str, ag_news_datamodule.text_pipeline)])