from .model import BertForCSC
import re
from transformers import BertTokenizer
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class CSCDataset(Dataset):
    def __init__(self, data):
        self.data = data
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        return {'input_ids': torch.tensor(self.data[index][0], dtype=torch.long), 
                'attention_mask': torch.tensor(self.data[index][1], dtype=torch.long)}

class CSCModel:
    def __init__(self, model_path, max_seq_len, batch_size) -> None:
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.model = BertForCSC.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.batch_size = batch_size

    def __strip(self, text):
        r = re.compile('\s')
        res = []
        sp_info = []
        for i in range(len(text)):
            if r.match(text[i]):
                sp_info.append((i, text[i]))
            else:
                res.append(text[i])
        return res, sp_info

    def __split(self, text, max_seq_len):
        span = []
        while len(text) > max_seq_len - 2:
            span.append(self.__tokenize(text[:max_seq_len-2]))
            text = text[max_seq_len-2:]
        span.append(self.__tokenize(text))
        return span

    def __tokenize(self, text):
        word = []
        for i in range(len(text)):
            word.extend(self.tokenizer.tokenize(text[i]))
        text =  ["[CLS]"] + word + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_len - len(input_ids))
        return input_ids + padding, input_mask + padding

    def __insert(self, text, sp_info):
        for idx, sp in sp_info:
            text.insert(idx, sp)
        return ''.join(text)

    def correct(self, text):
        sp_text, sp_info = self.__strip(text)
        blocks = self.__split(sp_text, self.max_seq_len)
        input_batch = DataLoader(CSCDataset(blocks), shuffle=False, batch_size=self.batch_size)
        predict_list = None
        mask_len = 0
        with torch.no_grad():
            for batch in input_batch:
                batch = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                predict = self.model(batch[0], batch[1])
                if predict_list:
                    predict_list = torch.cat((predict_list, predict[:, 1:-1]))
                else:
                    predict_list = predict[:, 1:-1]
                mask_len += torch.sum(batch[1]) - 2*batch[1].size(0)
        output = self.tokenizer.decode(torch.flatten(predict_list)[:mask_len]).split(' ')
        res = self.__insert(output, sp_info)
        output = [sp_text[i] if output[i]=='あ' else output[i] for i in range(len(output))]
        assert len(res) == len(text)
        return res
