import time
import yaml
import torch
from Data.dataset import *
from Data.utils import Batch
from Train.loss import LabelSmoothing, LossComputation
from Train.optimizer import NoamOpt
from transformer.model import Transformer


def single_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch = Batch(batch.src, batch.trg, PAD_IDX)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.n_tokens)

        total_loss += loss.numpy()
        total_tokens += batch.n_tokens.numpy()
        tokens += batch.n_tokens.numpy()
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss.numpy() / batch.n_tokens.numpy(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def main(opt):
    model = Transformer(opt)

    optimizer = NoamOpt(opt['d_model'], opt['optim_factor'], opt['n_warmup_steps'],
                        torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=(0.9, 0.98), eps=1e-9))

    criterion = LabelSmoothing(vocab_size=len(TRG.vocab), pad_idx=PAD_IDX, smoothing=opt['smoothing'])

    for epoch in range(option["max_epochs"]):
        model.train()
        single_epoch(train_iterator,
            model,
            LossComputation(criterion, optimizer=optimizer)
        )
        model.eval()
        loss = single_epoch(valid_iterator,
            model,
            LossComputation( criterion, optimizer=None)
        )
        print(loss)

    torch.save(model.state_dict(), opt['model_path'])



if __name__ == '__main__':
    f = open('../config.yaml', encoding='utf-8')
    option = yaml.safe_load(f)
    option['src_vocab_size'] = src_vocab_size
    option['trg_vocab_size'] = trg_vocab_size
    print(option)
    main(option)
    print('Terminated.')