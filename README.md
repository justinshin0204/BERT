# BERT Implementation
This repository is an attempt to implement **BERT** from scratch. The goal is to understand the inner workings of BERT by building it step-by-step, rather than relying on pre-built libraries. Through this process, we aim to gain a deeper understanding of its architecture and functionality.

# BERT
Before we start implementing, let's first discuss what BERT is.

BERT is a transformer-based model designed to pre-train deep bidirectional representations by considering the left and right context simultaneously in all layers using the transformer encoder architecture. 
Since its introduction in 2019, the BERT model has become the foundation for many subsequent models in the field of NLP.

The architecture is pretty similar to the original Transformer's encoder but it's slightly different.
Before we discuss the details, it's important to first understand how to train BERT, as knowing the training process is essential to understanding the modifications made to the architecture.
## Input/Output Representations
BERT receives two sentences merged into a single input sequence.
there is always a [CLS] token that represents the whole sentence. And each sentence is separated by a [SEP] token.
its form would be like " [CLS] + sentence1 +[SEP] + sentence2 + [SEP]"
![image](https://github.com/justinshin0204/BERT/assets/93083019/458a80ef-34d7-469a-9023-026586f90507)
We add **segment embeddings** to differentiate the sentences, and  **positional embeddings** to indicate the position of each token.

Concatenating makes more sense, but it's more resource-intensive.
## Pre-training
Now let's jump into two pre-training techniques
### Masked Language Modeling
We can intuitively think that bi-directional attention model would be much stronger , But since bi-directional conditioning would allow each word to indirectly “see itself” ( Because it provides more context ), We need an alternative approach.

So, the authors of the paper devised MLM.
![image](https://github.com/justinshin0204/BERT/assets/93083019/41bdfee9-fa85-4402-bbb9-88a6fc0970e3)
While using the MLM, we randomly choose tokens with a 15% probability. Among these, 80% are replaced with a mask token, 10% are replaced with a random token, and the remaining 10% are left unchanged.
( Since we don't use [MASK] in fine-tuning, we have to put some original sentence itself )

### NSP

Next Sentence Prediction (NSP) is another crucial task used during BERT's pre-training to understand the relationship between two sentences.
![image](https://github.com/justinshin0204/BERT/assets/93083019/7309bf78-94ba-44f9-9d78-385c11cab5ba)
The fully connected layer is attached to the hidden state of the [CLS] token and predicts if the second sentence is connected to the first one or not.


## Model architecture
![image](https://github.com/justinshin0204/BERT/assets/93083019/46c633a7-9690-4714-8899-33ab883e267c)
Now, let's discuss the model architecture: it consists of 12 layers, 12 attention heads, and a word dimension of 768.
Each block is the same as a Transformer encoder, but the final part differs due to the MLM and NSP tasks.

We've briefly discussed about BERT.
Now, lets move on to the implementation par




## Installation

To install the required datasets package, run the following command:

```bash
pip install transformers torch datasets
```
Import the necessary modules in your python script
```py
from transformers import BertTokenizer
from torch import nn, optim
from datasets import load_dataset
import random
from torch.nn.utils.rnn import pad_sequence
```

## Load BERT Tokenizer

To load the BERT tokenizer and obtain the special token IDs, you can use the following code:

```python
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get special token IDs
pad_idx = tokenizer.pad_token_id
mask_idx = tokenizer.mask_token_id
sep_idx = tokenizer.sep_token_id
cls_idx = tokenizer.cls_token_id

print("pad_idx =", pad_idx)
print("mask_idx =", mask_idx)
print("sep_idx =", sep_idx)
print("cls_idx =", cls_idx)
```

## Setting the hyperparameters


