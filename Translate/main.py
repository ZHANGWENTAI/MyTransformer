import yaml
import torch
from Data.dataset import *
from transformer.model import Transformer
from transformer.translator import Translator


def main(opt):

    model = Transformer(opt)
    checkpoint = torch.load(opt['model_path'])
    model.load_state_dict(checkpoint)

    translator = Translator(opt)

    for i, batch in enumerate(valid_iterator):
        print(i, end='\n')
        src = batch.src

        src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
        translator.translate_batch(src, src_mask)

        print("Target:", end="\t")
        for k in range(src.size(1)):
            for j in range(1, batch.trg.size(1)):
                sym = TRG.vocab.itos[batch.trg[k, j]]
                if sym == KEY_EOS: break
                print(sym, end=" ")
            print('\n')


if __name__ == '__main__':
    f = open('../config.yaml', encoding='utf-8')
    option = yaml.safe_load(f)
    option['src_vocab_size'] = src_vocab_size
    option['trg_vocab_size'] = trg_vocab_size
    print(option)
    main(option)
    print('Terminated')