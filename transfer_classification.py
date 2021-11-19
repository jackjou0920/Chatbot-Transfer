import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransferClassifier:
    def __init__(self, mode="cuda", model_path="model_classification_gpu_epoch_1_batch_64"):
        self.MAX_LEN = 70
        self.BATCH_SIZE = 64
        self.label_dict = {0: "OTHER", 1: "TRANSFER"}
        self.device = torch.device(mode)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = torch.load(model_path)

    def get_label(self, label):
        if (label not in self.label_dict):
            return ("no label")
        return (self.label_dict[label])

    def _convert_text_to_ids(self, tokenizer, text, max_len=70):
        if isinstance(text, str):
            tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True)
            input_ids = tokenized_text["input_ids"]
            token_type_ids = tokenized_text["token_type_ids"]
        elif isinstance(text, list):
            input_ids = []
            token_type_ids = []
            for t in text:
                tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
                input_ids.append(tokenized_text["input_ids"])
                token_type_ids.append(tokenized_text["token_type_ids"])
        else:
            print("Unexpected input")
        return input_ids, token_type_ids

    def _seq_padding(self, tokenizer, X):
        pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        if len(X) <= 1:
            return torch.tensor(X)
        L = [len(x) for x in X]
        ML = max(L)
        X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
        return X

    def predict(self, text):
        input_ids, token_type_ids = self._convert_text_to_ids(self.tokenizer, text, self.MAX_LEN)
        input_ids = self._seq_padding(self.tokenizer, input_ids)
        token_type_ids = self._seq_padding(self.tokenizer, token_type_ids)

        input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
        input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

        logits = output.logits.detach().cpu().numpy()
        return(np.argmax(logits, axis=1)[0])
