class NoamOpt:

    def __init__(self, d_model, factor, warmup, optim):
        self.optim = optim
        self._step = 0
        self.warmup = warmup
        self.factor = factor  # 用来调节学习率的整体大小
        self.d_model = d_model
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optim.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optim.step()

    def rate(self, step=None):
        """Implement `lrate` in the paper"""
        if step is None:
            step = self._step
        return self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))