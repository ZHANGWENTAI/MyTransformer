from Data.dataset import *
from transformer.model import Transformer

class Translator(object):
    """ Load with trained model and handel the beam search """
    def __init__(self, opt, use_cuda):
        self.opt = opt
        self.use_cuda = use_cuda
        self.tt = torch.cuda if use_cuda else torch

        model = Transformer(opt)
        if use_cuda:
            print('Using GPU..')
            model = model.cuda()

        checkpoint = torch.load(opt['model_path'])
        model.load_state_dict(checkpoint)
        print('Loaded pre-trained model_state..')

        self.model = model
        self.model.eval()

    def translate_batch(self, src, src_mask):
        """TO DO"""
        pass

