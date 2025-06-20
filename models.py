import torch
import torch.nn as nn
import torch.nn.functional as F

class RealFFN(nn.Module):
    def __init__(self, d_in, d_hid, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LookupFFN(nn.Module):
    def __init__(self, d_in, d_out, num_buckets=256):
        super().__init__()
        self.num_buckets = num_buckets
        self.d_in = d_in
        self.d_out = d_out
        self.register_buffer('R', torch.randn(num_buckets, d_in))
        self.L = nn.Parameter(torch.randn(num_buckets, d_out))
    def forward(self, x):
        proj = torch.matmul(x, self.R.t())
        idx = torch.argmax(proj, dim=1)
        return self.L[idx]

class ImprovedLookupFFN(nn.Module):
    def __init__(self, d_in, num_classes, num_buckets=512, temperature=0.5):
        super().__init__()
        self.num_buckets = num_buckets
        self.d_in = d_in
        self.d_out = num_classes
        self.R = nn.Parameter(torch.randn(num_buckets, d_in))  # Learnable
        self.L = nn.Parameter(torch.randn(num_buckets, num_classes))
        self.temperature = temperature
        self.post = nn.Linear(num_classes, num_classes)
    def forward(self, x):
        proj = torch.matmul(x, self.R.t()) / self.temperature  # [batch, num_buckets]
        weights = F.softmax(proj, dim=1)  # [batch, num_buckets]
        out = torch.matmul(weights, self.L)  # [batch, num_classes]
        out = self.post(out)
        return out

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, ffn_type='real', num_classes=2, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if ffn_type == 'real':
            self.ffn = RealFFN(emb_dim, kwargs.get('d_hid', 128), num_classes)
        else:
            self.ffn = ImprovedLookupFFN(
                emb_dim,
                num_classes,
                kwargs.get('num_buckets', 512),
                kwargs.get('temperature', 0.5)
            )
    def forward(self, x):
        emb = self.embedding(x).mean(dim=1)
        return self.ffn(emb)