import torch

class Rnn_layer:
    def __init__(self, vocab_size, hidden_size):
        self.w = [
            torch.randn(hidden_size, hidden_size, requires_grad=True),  # wh
            torch.randn(vocab_size, hidden_size, requires_grad=True),   # wx
            torch.randn(1, hidden_size, requires_grad=True)             # bh
        ]
        self.ht = torch.zeros(1, hidden_size)

    def hidden_state(self, xt):
        wh, wx,bh = self.w
        self.ht = torch.tanh(self.ht @ wh + xt @ wx + bh)
        return self.ht
    


class LStm:
    def __init__(self, vocab_size, hidden_size):
        self.w = [
            torch.randn(vocab_size, hidden_size, requires_grad=True),  # wx_i
            torch.randn(hidden_size, hidden_size, requires_grad=True), # wh_i
            torch.randn(1, hidden_size, requires_grad=True),           # bi

            torch.randn(vocab_size, hidden_size, requires_grad=True),  # wx_f
            torch.randn(hidden_size, hidden_size, requires_grad=True), # wh_f
            torch.randn(1, hidden_size, requires_grad=True),           # bf

            torch.randn(vocab_size, hidden_size, requires_grad=True),  # wx_c
            torch.randn(hidden_size, hidden_size, requires_grad=True), # wh_c
            torch.randn(1, hidden_size, requires_grad=True),           # bc

            torch.randn(vocab_size, hidden_size, requires_grad=True),  # wx_o
            torch.randn(hidden_size, hidden_size, requires_grad=True), # wh_o
            torch.randn(1, hidden_size, requires_grad=True),            # bo
        ]
        self.ht = torch.zeros(1, hidden_size)
        self.ct = torch.zeros(1, hidden_size)

    def hidden_state(self, xt):
        w = self.w
        it = torch.sigmoid(xt @ w[0] + self.ht @ w[1] + w[2])
        ft = torch.sigmoid(xt @ w[3] + self.ht @ w[4] + w[5])
        ct_bar = torch.tanh(xt @ w[6] + self.ht @ w[7] + w[8])
        ot = torch.sigmoid(xt @ w[9] + self.ht @ w[10] + w[11])

        self.ct = ft * self.ct + it * ct_bar
        self.ht = ot * torch.tanh(self.ct)
        return self.ht

class Gru:
    def __init__(self, vocab_size, hidden_size):
        self.w = [
            torch.randn(vocab_size, hidden_size, requires_grad=True),  # Wrx
            torch.randn(hidden_size, hidden_size, requires_grad=True), # Ur
            torch.randn(1, hidden_size, requires_grad=True),           # br

            torch.randn(vocab_size, hidden_size, requires_grad=True),  # Wrz
            torch.randn(hidden_size, hidden_size, requires_grad=True), # Uz
            torch.randn(1, hidden_size, requires_grad=True),           # bz

            torch.randn(vocab_size, hidden_size, requires_grad=True),  # Wx_h
            torch.randn(hidden_size, hidden_size, requires_grad=True), # Wh_h
            torch.randn(1, hidden_size, requires_grad=True),          # bh

        ]
        self.ht = torch.zeros(1, hidden_size)

    def hidden_state(self, xt):
        w = self.w
        rt = torch.sigmoid(xt @ w[0] + self.ht @ w[1] + w[2])
        zt = torch.sigmoid(xt @ w[3] + self.ht @ w[4] + w[5])
        ht_bar = torch.tanh(xt @ w[6] + (rt * self.ht) @ w[7] + w[8])
        self.ht = zt * self.ht + (1 - zt) * ht_bar
        return self.ht


class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = torch.randn(vocab_size, d_model, requires_grad=True)
    
    def get_embedings(self, x):
        return self.embeddings[x]