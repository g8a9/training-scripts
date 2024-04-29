import fire
from tqdm import tqdm
import time
import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)

# Inspiration from https://github.com/huggingface/peft/blob/main/examples/sequence_classification/LoRA.ipynb
class NeutralScorer:
    def __init__(self, model_name_or_path: str, device="cuda", torch_dtype=torch.bfloat16, use_lora:bool=False):
        self.device = device

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")

        self.model.config.id2label = {0: "neutral", 1: "gendered"}

        if not hasattr(self.tokenizer, "pad_token_id"):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if use_lora:
            self.config = PeftConfig.from_pretrained(model_name_or_path)
            self.model = PeftModel.from_pretrained(
                self.model, model_name_or_path
            )

        self.model.to(device).eval()

    @inference_decorator()
    def predict(self, texts, batch_size=4, num_workers=0):
        data = Dataset.from_dict({"text": texts})
        data = data.map(
            lambda x: self.tokenizer(x["text"], truncation=True),
            batched=True,
            remove_columns=["text"],
        )
        collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

        final_preds = list()
        for step, batch in tqdm(enumerate(loader), desc="Batch", total=len(texts) // batch_size):
            batch.to(self.device)
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions = [self.model.config.id2label[i.item()] for i in predictions]
            final_preds.extend(predictions)

        return final_preds

def main(model_name_or_path: str):

    scorer = NeutralScorer(model_name_or_path)

    def read_sents(filename):
        with open(filename) as fp:
            lines = [f.strip() for f in fp.readlines()]
        return lines

    for system in ["Amazon", "DeepL"]:
        set_g = read_sents(f"./data/GNT-output/{system}-G-original")
        set_pe1 = read_sents(f"./data/GNT-output/{system}-N-PEbyTransl1")
        set_pe2 = read_sents(f"./data/GNT-output/{system}-N-PEbyTransl2")
        set_pe3 = read_sents(f"./data/GNT-output/{system}-N-PEbyTransl3")

        set_n = set_pe1 + set_pe2 + set_pe3

        y_true = ["gendered"] * len(set_g) + ["neutral"] * len(set_n)
        print("y true", y_true[:5])
        y_pred = scorer.predict(set_g) + scorer.predict(set_n)
        print("y pred", y_pred[:5])

        print(f"\n#### STATS for {system} ####\n")
        print("F1 (Macro)", f1_score(y_true, y_pred, average="macro"))
        print("Overall Accuracy", accuracy_score(y_true, y_pred))
        print("Accuracy (Set-G)", accuracy_score(y_true[:len(set_g)], y_pred[:len(set_g)]))
        print("Accuracy (Set-N)", accuracy_score(y_true[len(set_g):], y_pred[len(set_g):]))
        print("Classification report", classification_report(y_true, y_pred))
        print(f"\n####\n")


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed {time.time() - stime} seconds.")
