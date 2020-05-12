import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, x, target):
        """
        :param x: [batch_size, trg_seq_len - 1, trg_vocab_size]
        :param target: [batch_size, trg_seq_len - 1]
        :return:
        """
        assert x.size(-1) == self.vocab_size
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 而不是 -1 是因为 <pad> 的概率为 0

        true_dist.scatter_(-1, target.data.unsqueeze(-1), self.confidence)

        true_dist[:, :, self.pad_idx] = 0
        true_dist.requires_grad_(False)

        return self.criterion(x, true_dist)


class LossComputation:

    def __init__(self, criterion, optimizer=None):
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, label, n_token):
        loss = self.criterion(x, label) / n_token
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optim.zero_grad()
        return loss.item() * n_token.float()