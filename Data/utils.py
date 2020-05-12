import torch

'''get mask from the batch'''
def get_subsequent_mask(d_model):
    return torch.ones(1, d_model, d_model) - torch.triu(torch.ones((1, d_model, d_model)), diagonal=1)


def get_trg_mask(trg, pad_idx):
    """Create a mask to hide padding and future words."""
    trg_mask = (trg != pad_idx).unsqueeze(-2)
    trg_mask = trg_mask & get_subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    return trg_mask


class Batch:
    def __init__(self, src, trg, pad_idx=1):
        self.src = src
        self.src_mask = (src != pad_idx).unsqueeze(-2)
        self.trg = trg[:, :-1]
        self.trg_y = trg[:, 1:]

        self.trg_mask = get_trg_mask(self.trg, pad_idx)
        self.n_tokens = (self.trg_y != pad_idx).sum()  # 记录字符数

