import torch
import torch.nn as nn


class Dnn(nn.Module):
    def __init__(self, embed_mat, seq_len):
        super(Dnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.encode1 = nn.Sequential(nn.Linear(self.embed_len, 200),
                                     nn.ReLU())
        self.encode2 = nn.Sequential(nn.Linear(200, 200),
                                     nn.ReLU())
        self.match1 = nn.Sequential(nn.Linear(800, 200),
                                    nn.ReLU())
        self.match2 = nn.Sequential(nn.Dropout(0.2),
                                    nn.Linear(200, 1))

    def forward(self, x, y):
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.encode1(x)
        x = self.encode2(x)
        y = self.embed(y)
        y = torch.mean(y, dim=1)
        y = self.encode1(y)
        y = self.encode2(y)
        diff = torch.abs(x - y)
        prod = x * y
        z = torch.cat([x, y, diff, prod], dim=1)
        z = self.match1(z)
        return self.match2(z)


class DnnEncode(nn.Module):
    def __init__(self, embed_mat, seq_len):
        super(DnnEncode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.encode1 = nn.Sequential(nn.Linear(self.embed_len, 200),
                                     nn.ReLU())
        self.encode2 = nn.Sequential(nn.Linear(200, 200),
                                     nn.ReLU())

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.encode1(x)
        return self.encode2(x)


class Cnn(nn.Module):
    def __init__(self, embed_mat, seq_len):
        super(Cnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.cap1 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=1, padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.cap2 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=2, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len + 1))
        self.cap3 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.encode = nn.Sequential(nn.Linear(192, 200),
                                    nn.ReLU())
        self.match1 = nn.Sequential(nn.Linear(800, 200),
                                    nn.ReLU())
        self.match2 = nn.Sequential(nn.Dropout(0.2),
                                    nn.Linear(200, 1))

    def forward(self, x, y):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x1 = self.cap1(x)
        x2 = self.cap2(x)
        x3 = self.cap3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(x.size(0), -1)
        x = self.encode(x)
        y = self.embed(y)
        y = y.permute(0, 2, 1)
        y1 = self.cap1(y)
        y2 = self.cap2(y)
        y3 = self.cap3(y)
        y = torch.cat((y1, y2, y3), dim=1)
        y = y.view(y.size(0), -1)
        y = self.encode(y)
        diff = torch.abs(x - y)
        prod = x * y
        z = torch.cat([x, y, diff, prod], dim=1)
        z = self.match1(z)
        return self.match2(z)


class CnnEncode(nn.Module):
    def __init__(self, embed_mat, seq_len):
        super(CnnEncode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.cap1 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=1, padding=0),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.cap2 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=2, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len + 1))
        self.cap3 = nn.Sequential(nn.Conv1d(self.embed_len, 64, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(seq_len))
        self.encode = nn.Sequential(nn.Linear(192, 200),
                                    nn.ReLU())

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x1 = self.cap1(x)
        x2 = self.cap2(x)
        x3 = self.cap3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(x.size(0), -1)
        return self.encode(x)


class Rnn(nn.Module):
    def __init__(self, embed_mat, seq_len):
        super(Rnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.encode = nn.LSTM(self.embed_len, 200, batch_first=True)
        self.match1 = nn.Sequential(nn.Linear(800, 200),
                                    nn.ReLU())
        self.match2 = nn.Sequential(nn.Dropout(0.2),
                                    nn.Linear(200, 1))

    def forward(self, x, y):
        x = self.embed(x)
        x, h_n = self.encode(x)
        x = x[:, -1, :]
        y = self.embed(y)
        y, h_n = self.encode(y)
        y = y[:, -1, :]
        diff = torch.abs(x - y)
        prod = x * y
        z = torch.cat([x, y, diff, prod], dim=1)
        z = self.match1(z)
        return self.match2(z)


class RnnEncode(nn.Module):
    def __init__(self, embed_mat, seq_len):
        super(RnnEncode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.seq_len = seq_len
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.encode = nn.LSTM(self.embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        x, h_n = self.encode(x)
        return x[:, -1, :]


class Match(nn.Module):
    def __init__(self):
        super(Match, self).__init__()
        self.match1 = nn.Sequential(nn.Linear(800, 200),
                                    nn.ReLU())
        self.match2 = nn.Sequential(nn.Dropout(0.2),
                                    nn.Linear(200, 1))

    def forward(self, x, y):
        diff = torch.abs(x - y)
        prod = x * y
        z = torch.cat([x, y, diff, prod], dim=1)
        z = self.match1(z)
        return self.match2(z)
