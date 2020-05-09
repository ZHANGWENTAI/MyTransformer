from Data.dataset import *
from Data.utils import get_subsequent_mask

def greedy_decode(model, src, src_mask):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(SOS_IDX).type_as(src.data)
    for i in range(src.size(1) - 1):
        out = model.decode(memory, src_mask,
                           ys,
                           get_subsequent_mask(ys.size(1))
                                    .type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).fill_(next_word).type_as(src.data)], dim=1)
    return ys



