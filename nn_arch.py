import torch
import torch.nn as nn


seq_len = 30


class Dnn(nn.Module):
    def __init__(self, embed_mat):
        super(Dnn, self).__init__()
        self.encode = DnnEncode(embed_mat)
        self.match = Match('dnn')

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        return self.match(x, y)


class DnnEncode(nn.Module):
    def __init__(self, embed_mat):
        super(DnnEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.la1 = nn.Sequential(nn.Linear(embed_len, 200),
                                 nn.ReLU())
        self.la2 = nn.Sequential(nn.Linear(200, 200),
                                 nn.ReLU())

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.la1(x)
        return self.la2(x)


class Cnn(nn.Module):
    def __init__(self, embed_mat):
        super(Cnn, self).__init__()
        self.encode = CnnEncode(embed_mat)
        self.match = Match('cnn')

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        return self.match(x, y)


class CnnEncode(nn.Module):
    def __init__(self, embed_mat):
        super(CnnEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.cap1 = nn.Sequential(nn.Conv1d(embed_len, 64, kernel_size=1, padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.cap2 = nn.Sequential(nn.Conv1d(embed_len, 64, kernel_size=2, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len + 1))
        self.cap3 = nn.Sequential(nn.Conv1d(embed_len, 64, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.la = nn.Sequential(nn.Linear(192, 200),
                                nn.ReLU())

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x1 = self.cap1(x)
        x2 = self.cap2(x)
        x3 = self.cap3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(x.size(0), -1)
        return self.la(x)


class Rnn(nn.Module):
    def __init__(self, embed_mat):
        super(Rnn, self).__init__()
        self.encode = RnnEncode(embed_mat)
        self.match = Match('rnn')

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        return self.match(x, y)


class RnnEncode(nn.Module):
    def __init__(self, embed_mat):
        super(RnnEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.ra = nn.LSTM(embed_len, 200, batch_first=True, bidirectional=True)
        self.mp = nn.MaxPool1d(seq_len)

    def forward(self, x):
        x = self.embed(x)
        h, hc_n = self.ra(x)
        h = h.permute(0, 2, 1)
        x = self.mp(h)
        return x.view(x.size(0), -1)


class Match(nn.Module):
    def __init__(self, name):
        super(Match, self).__init__()
        feat_len = 1600 if name == 'rnn' else 800
        self.la = nn.Sequential(nn.Linear(feat_len, 200),
                                nn.ReLU())
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, 1))

    def forward(self, x, y):
        diff = torch.abs(x - y)
        prod = x * y
        z = torch.cat([x, y, diff, prod], dim=1)
        z = self.la(z)
        return self.dl(z)
