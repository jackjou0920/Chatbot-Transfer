import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransferNer:
    def __init__(self, model_path="model_ner_gpu_epoch_2_batch_64"):
        self.MAX_LEN = 30
        self.BATCH_SIZE = 64
        self.tags_vals = ['TransFr_B', 'TransFr_I', 'TransTo_B', 'TransTo_I', 'AMOUNT_B', 'BANK_B', 'BANK_I', 'O']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = torch.load(model_path)

    def _text_to_dataloader(self, tokenized_test_texts):
        input_ids = pad_sequences(
            [self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_test_texts],
            maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post"
        )
        test_attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

        test_inputs = torch.tensor(input_ids)
        test_masks = torch.tensor(test_attention_masks)

        test_sentence_data = TensorDataset(test_inputs, test_masks)
        train_sentence_sampler = RandomSampler(test_sentence_data)
        test_sentence_dataloader = DataLoader(
            test_sentence_data, sampler=train_sentence_sampler, batch_size=self.BATCH_SIZE
        )
        return (test_sentence_dataloader)

    def _tag_to_ner_dict(self, token, token_tag):
        trans_to, trans_from, bank = "", "", ""
        ner_dict = dict({"TransFr": [], "TransTo": [], "BANK": [], "AMOUNT": []})

        for i in range(len(token_tag)):
            if (token_tag[i] == "TransTo_B" or token_tag[i] == "TransTo_I"):
                trans_to += token[i]
                if (i+1 >= len(token_tag) or token_tag[i+1] != "TransTo_I"):
                    if (trans_to != ""):
                        ner_dict["TransTo"].append(trans_to)
                    trans_to = ""
            elif (token_tag[i] == "TransFr_B" or token_tag[i] == "TransFr_I"):
                trans_from += token[i]
                if (i+1 >= len(token_tag) or token_tag[i+1] != "TransFr_I"):
                    if (trans_from != ""):
                        ner_dict["TransFr"].append(trans_from)
                    trans_from = ""
            elif (token_tag[i] == "BANK_B" or token_tag[i] == "BANK_I"):
                bank += token[i]
                if (i+1 >= len(token_tag) or token_tag[i+1] != "BANK_I"):
                    if (bank != ""):
                        ner_dict["BANK"].append(bank)
                    bank = ""
            elif (token_tag[i] == "AMOUNT_B"):
                ner_dict["AMOUNT"].append(token[i])
        return (ner_dict)

    def predict(self, text):
        tokenized_test_texts = [self.tokenizer.tokenize(sent) for sent in text]

        # predict tokenized tag
        predictions = list()
        for batch in self._text_to_dataloader(tokenized_test_texts):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            self.model.eval()
            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
            logits = logits.detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        pred_tags = [[self.tags_vals[p_i] for p_i in p] for p in predictions]
        
        print(tokenized_test_texts[0])
        print(pred_tags[0])
        
        # concate to token tag
        token = list()
        for tok in tokenized_test_texts[0]:
            if ('##' in tok):
                token[-1] = token[-1] + tok.replace('##', '')
            else:
                token.append(tok)
        token_tag = pred_tags[0][:len(token)]

        return (self._tag_to_ner_dict(token, token_tag))
