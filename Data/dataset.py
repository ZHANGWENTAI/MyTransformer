from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

KEY_EOS = '<eos>'
KEY_SOS = '<sos>'
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SRC = Field(tokenize="spacy",
            tokenizer_language="de_core_news_sm",
            init_token=KEY_SOS,
            eos_token=KEY_EOS,
            batch_first=True)
TRG = Field(tokenize="spacy",
            tokenizer_language="en_core_web_sm",
            init_token=KEY_SOS,
            eos_token=KEY_EOS,
            batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG), path='../.data/multi30k/')

print('train_data:', len(train_data))
print('valid_data:', len(valid_data))
print('test_data:', len(test_data))

SRC.build_vocab(train_data.src, min_freq = 2) # 出现频率小于2次的视为 <unk>
TRG.build_vocab(train_data.trg, min_freq = 2)

src_vocab_size = len(SRC.vocab)
trg_vocab_size = len(TRG.vocab)

print('德语词汇表大小：', len(SRC.vocab), ',前20个：', SRC.vocab.itos[0:20])
print('英语词汇表大小：', len(TRG.vocab), ',前20个：', TRG.vocab.itos[0:20])

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = 256)

