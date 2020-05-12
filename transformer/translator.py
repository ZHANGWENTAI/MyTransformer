import torch
from Data.dataset import *
from Data.utils import get_trg_mask
from transformer.model import Transformer

class Translator(object):
    """ Load with trained model and handel the beam search """
    def __init__(self, opt):
        self.opt = opt

        model = Transformer(opt)

        checkpoint = torch.load(opt['model_path'])
        model.load_state_dict(checkpoint)
        print('Loaded pre-trained model_state..')

        self.model = model
        self.model.eval()

    def translate_batch(self, src, src_mask):
        """
        :param src: [batch_size, seq_len]
        :param src_mask: [batch_size, seq_len]
        """
        opt = self.opt
        model = self.model
        batch_size = src.size(0)
        beam_size = opt['beam_size']
        memory = model.encode(src, src_mask)

        '''initial batch of target sequence'''
        beams = torch.zeros(batch_size, beam_size, 1, dtype=torch.long).fill_(SOS_IDX)
        '''score of each beam'''
        curr_scores = torch.ones(batch_size, beam_size)

        for step in range(opt['max_trg_len']):
            print('step:', step)
            alter_path = []
            '''current scores waited to be sorted'''
            all_scores = []

            for i in range(beam_size):
                '''[batch_size, seq_len, d_model]'''
                decode_output = model.decode(memory, src_mask, beams[:, i, :] , get_trg_mask(beams[:, i, :], PAD_IDX))
                '''[batch_size, seq_len, vocab_size]'''
                proj_output = model.project(decode_output)
                '''[batch_size, beam_size]'''
                scores, word_idx = torch.topk(proj_output[:, -1, :], k=beam_size, dim=-1)

                '''alter_path: [beam_size, batch_size, beam_size, seq_len + 1]'''
                alter_path.append(
                    torch.cat(
                        (beams[:, i, :].repeat(1, beam_size), word_idx),
                        dim=-1
                    )
                )

                '''[batch_size, beam_size]'''
                all_scores.append(
                    torch.mul(curr_scores[:, i].unsqueeze(-1), scores)
                )

            all_scores = torch.stack(all_scores).transpose(0, 1).contiguous().view(batch_size, -1)
            '''[batch_size, beam_size]'''
            curr_scores, seq_idx = torch.topk(all_scores, beam_size, -1)

            '''[batch_size, beam_size * beam_size, seq_len'''
            alter_path = torch.stack(alter_path).view(batch_size, beam_size * beam_size, -1)

            '''update beam'''
            beams = torch.cat((beams, torch.zeros(batch_size, beam_size, 1, dtype=torch.long)), dim=-1)
            for i in range(batch_size):
                beams[i] = torch.index_select(alter_path[i], 0, seq_idx[i])

        with open(opt['translate_output'], 'a',) as f:
            for i in range(batch_size):
                for j in range(1, beams.size(-1)):
                    sym = TRG.vocab.itos[beams[i, 0, j]]
                    if sym == KEY_EOS: break
                    f.write(sym + " ")
                f.write('\n')

        f.close()








