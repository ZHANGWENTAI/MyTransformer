# MyTransformer
Pytorch implement of [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Configuration
you can rewrite the file config.yaml to configure your training process or translation process.

### Data
Using Multi30k dataset from torchtext, about how to use the dataset, [this](https://pytorch.org/text/data.html#) will help you.

### Model
all details about implementation are in model.ipynb.

### To_Do
- It takes too long to translate long sentences. Optimize the beam search algorithm in Model/translator, the method `translate_batch`.
- Surrport cuda
