import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset

class SimpleTokenizer:
    def __init__(self, texts):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
    def encode(self, text):
        return [self.vocab.get(word, 1) for word in text.split()]
    def decode(self, token_ids):
        return " ".join([self.rev_vocab.get(idx, "[UNK]") for idx in token_ids])

class SST2Dataset(Dataset):
    def __init__(self, split, tokenizer, max_len=32):
        ds = load_dataset("stanfordnlp/sst2")
        self.texts = ds[split]['sentence']
        self.labels = ds[split]['label']
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

class YahooAnswersDataset(Dataset):
    def __init__(self, split_indices, ds, tokenizer, max_len=64):
        self.texts = [ds['question'][i] for i in split_indices]
        self.labels = [ds['answer'][i] for i in split_indices]
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

class IMDBDataset(Dataset):
    def __init__(self, split, tokenizer, max_len=256):
        ds = load_dataset("imdb")
        self.texts = ds[split]['text']
        self.labels = ds[split]['label']
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

def get_dataloaders(batch_size=64, max_len=32):
    ds = load_dataset("stanfordnlp/sst2")
    tokenizer = SimpleTokenizer(ds['train']['sentence'])
    train_dataset = SST2Dataset('train', tokenizer, max_len)
    test_dataset = SST2Dataset('validation', tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader, tokenizer

def get_dataloaders_yahoo(batch_size=64, max_len=64, val_frac=0.1):
    ds = load_dataset("sentence-transformers/yahoo-answers", "question-answer-pair")['train']
    print(f"Yahoo Answers columns: {ds.column_names}")
    tokenizer = SimpleTokenizer(ds['question'])
    n = len(ds)
    n_val = int(n * val_frac)
    n_train = n - n_val
    indices = list(range(n))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    train_dataset = YahooAnswersDataset(train_indices, ds, tokenizer, max_len)
    val_dataset = YahooAnswersDataset(val_indices, ds, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, tokenizer

def get_dataloaders_imdb(batch_size=64, max_len=256):
    ds = load_dataset("imdb")
    tokenizer = SimpleTokenizer(ds['train']['text'])
    train_dataset = IMDBDataset('train', tokenizer, max_len)
    test_dataset = IMDBDataset('test', tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader, tokenizer 