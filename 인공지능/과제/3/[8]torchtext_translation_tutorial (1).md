
`TorchText`와 함께하는 언어 번역
===================================

`torchtext`를 활용해 영어/독일어 데이터셋을 전처리 하고, sequence-to-sequence 모델을 학습시켜, 독일어를 영어로 번역할 수 있는 모델을 만드는 것이 목표

### `Field` and `TranslationDataset`
`torchtext`의 유틸리티를 활용해 데이터 전처리를 진행한다

- `Field` 클래스
  - 각각의 문장이 어떻게 전처리 되어야 하는지 정의
- `TranslationDataset` 클래스

데이터셋은 `Multi30K` 데이터셋을 사용한다(30000 여개의 문장 포함, 영어/독일어)


```python
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))
```

    downloading training.tar.gz


    training.tar.gz: 100%|██████████| 1.21M/1.21M [00:02<00:00, 479kB/s]


    downloading validation.tar.gz


    validation.tar.gz: 100%|██████████| 46.3k/46.3k [00:00<00:00, 175kB/s]


    downloading mmt_task1_test2016.tar.gz


    mmt_task1_test2016.tar.gz: 100%|██████████| 66.2k/66.2k [00:00<00:00, 166kB/s]

`torchtext` `Field` 클래스의 `build_vocab()`함수를 활용해 각각의 언어에 맞는 Vocabulary를 만들 수 있다


```python
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
```

이 라인들이 실행 된 뒤

- `SRC.vocab.stoi`는 생성된 token들이 각각 key로, 해당되는 indice가 value로 저장되는 딕셔너리가 된다
- `SRC.vocab.itos`는 같은 딕셔너리지만, key와 value가 바뀐 딕셔너리가 된다

### ``BucketIterator``
`torchtext`의 `BucketIterator`는 `TranslationDataset`을 첫 인자로 받아, 비슷한 길이의 데이터들을 배치로 묶고, 패딩의 길이를 최소화 하되, 매 epoch마다 새롭게 뒤섞인 배치를 만들 수 있도록 해준다.


```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)
```

위를 실행한 뒤 `for i, batch in enumerate(iterator)` 반복문에서 
`batch`는 `src`와 `trg` 속성을 가지게 되며, 손쉽게 호출이 가능해진다

### 신경망과 최적화 기법 정의
![img](https://github.com/SethHWeidman/pytorch-seq2seq/raw/20828f64e20d2638c1bc3348e6ae6614e2c91b83/assets/seq2seq7.png)

- 위와 같은 Encoder-Decoder 구조를 가지는 신경망을 만들 것이다


```python
import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters())


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
```

    The model has 1,856,685 trainable parameters

손실함수는 교차엔트로피 손실함수를 사용한다

- 성능을 평가할 때는 손실함수를 비활성화 시킬 것을 명심!


```python
PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

- 이제 모델의 학습과 평가가 가능하다


```python
import math
import time


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
```

    Epoch: 01 | Time: 0m 28s
    	Train Loss: 5.679 | Train PPL: 292.619
    	 Val. Loss: 5.251 |  Val. PPL: 190.821
    Epoch: 02 | Time: 0m 28s
    	Train Loss: 5.015 | Train PPL: 150.606
    	 Val. Loss: 5.043 |  Val. PPL: 154.873
    Epoch: 03 | Time: 0m 28s
    	Train Loss: 4.673 | Train PPL: 107.065
    	 Val. Loss: 4.813 |  Val. PPL: 123.128
    Epoch: 04 | Time: 0m 28s
    	Train Loss: 4.478 | Train PPL:  88.015
    	 Val. Loss: 4.790 |  Val. PPL: 120.268
    Epoch: 05 | Time: 0m 28s
    	Train Loss: 4.358 | Train PPL:  78.100
    	 Val. Loss: 4.791 |  Val. PPL: 120.379
    Epoch: 06 | Time: 0m 28s
    	Train Loss: 4.268 | Train PPL:  71.368
    	 Val. Loss: 4.740 |  Val. PPL: 114.488
    Epoch: 07 | Time: 0m 27s
    	Train Loss: 4.179 | Train PPL:  65.312
    	 Val. Loss: 4.678 |  Val. PPL: 107.588
    Epoch: 08 | Time: 0m 28s
    	Train Loss: 4.074 | Train PPL:  58.771
    	 Val. Loss: 4.539 |  Val. PPL:  93.564
    Epoch: 09 | Time: 0m 28s
    	Train Loss: 3.981 | Train PPL:  53.588
    	 Val. Loss: 4.481 |  Val. PPL:  88.282
    Epoch: 10 | Time: 0m 28s
    	Train Loss: 3.905 | Train PPL:  49.642
    	 Val. Loss: 4.404 |  Val. PPL:  81.809
    | Test Loss: 4.418 | Test PPL:  82.945 |

- 10 세대를 거쳐 학습이 되었다