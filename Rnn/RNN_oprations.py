import string
import torch

def tri_gram_vocab():
    alphabets = string.ascii_lowercase
    indices = torch.randperm(len(alphabets)).tolist()
    return {f"{alphabets[i]}": indices[i] for i in range(len(indices)) }

def tokenize(vocabulary, target):
    tokenization = []
    for char in target:
        tokenization.append(vocabulary[char])
    return tokenization
def get_xy(seq):
    x = seq[:-1]
    y = seq[1:]
    return x, y

def one_hot_encoding(vocab_size, index):
    vec = torch.zeros(vocab_size)
    vec[index] = 1
    return vec
def get_embedings(vocabulaire,len_embeding ):
 embeding_mat=torch.randn(len(vocabulaire),len_embeding)
 return embeding_mat

def postional_embeding(embeding,position):
    positional=[]
    i=0
    for pos in range(len(embeding)):
        if pos % 2==0:
            value=torch.sin(torch.tensor(position/(10000**(2*int(i)/len(embeding)))))
        else:
            value=torch.cos(torch.tensor(position/(10000**(2*int(i)/len(embeding)))))
            i+=0.5
        positional.append(value)
    return torch.stack(positional)+torch.tensor(embeding)

