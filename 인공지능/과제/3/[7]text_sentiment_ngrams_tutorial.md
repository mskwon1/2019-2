## 텍스트 분류 튜토리얼

``TextClassification`` 데이터셋을 활용해 지도 학습 수행

### `ngrams`로 데이터 불러오기

`ngrams`를 통해 word를 하나 단위로만이 아니라 인접한 n개 word의 조합으로도 볼 수 있다
여기서는 `ngrams`를 2로 설정해 word 하나, word 2개의 조합으로 text를 만든다

```python
import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
if not os.path.isdir('./.data'):
	os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

    ag_news_csv.tar.gz: 11.8MB [00:00, 27.2MB/s]
    120000lines [00:11, 10794.80lines/s]
    120000lines [00:23, 5200.79lines/s]
    7600lines [00:01, 4923.44lines/s]


### 모델 정의하기

![](https://github.com/pytorch/tutorials/blob/gh-pages/_static/img/text_sentiment_ngrams_model.png?raw=1)

신경망은 위와 같은 구성을 가지며,

- `EmbeddingBag` layer와 linear layer로 구성되어 있다
- `EmbeddingBag` 레이어에서 Bag의 평균값을 계산한다
- Text Entry들은 길이가 다 다르지만, offset에 텍스트 길이가 저장되어 있어서 괜찮다
- 추가적으로, `EmbeddingBag`은 평균값들을 누적시켜 성능을 향상 시키고 메모리 효율성을 높인다


```python
import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
```

### 인스턴스 초기화 

The AG_NEWS dataset has four labels and therefore the number of classes
is four.

`AG_NEWS` 데이터셋에는 4개의 레이블이 있으며, 따라서 클래스도 4개가 된다

1. World
2. Sports
3. Business
4. Sci/Tec 

`VOCAB_SIZE`는 각 `vocab`의 길이에 맞게 초기화하고, `NUM_CLASS`는 label의 갯수(4개)에 맞게 초기화한다

```python
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
```

### 배치 관련 함수

텍스트 엔트리마다 길이가 다르기 때문에, `generate_batch()` 함수를 정의해 batch와 offset을 만든다

- `torch.utils.data.DataLoader`의 `collate_fn`파라미터로 해당 함수를 보내주면, mini-batch로 돌려준다.

텍스트 엔트리는 `EmbeddingBag`에 들어가기 전, 하나의 텐서로 포장된다

- offset은 각각 sequence의 시작 index를 표현한다
- label은 각각의 text entry의 label을 표현한다

```python
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
    
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label
```

### 훈련 / 평가를 위한 함수

`DataLoader`를 활용해 AG_NEWS 데이터셋을 불러와서 모델로 보내 학습 및 평가를 진행한다

```python
from torch.utils.data import DataLoader

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()
    
    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)
```

### 데이터셋을 나누고, 모델 실행

- AG_NEWS 데이터셋에는 검증 데이터셋이 없기 때문에, 나눠준다 0.95(훈련) / 0.05(검증)의 비율로 나눈다

- 손실함수는 교차엔트로피손실함수를 사용한다

- 최적화는 SGD 방식으로 진행한다

- Learning Rate 조절은 `lr_scheduler.StepLR`을 통해 진행한다

```python
import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
```

    Epoch: 1  | time in 0 minutes, 12 seconds
    	Loss: 0.0260(train)	|	Acc: 84.7%(train)
    	Loss: 0.0001(valid)	|	Acc: 88.0%(valid)
    Epoch: 2  | time in 0 minutes, 11 seconds
    	Loss: 0.0118(train)	|	Acc: 93.7%(train)
    	Loss: 0.0001(valid)	|	Acc: 90.5%(valid)
    Epoch: 3  | time in 0 minutes, 11 seconds
    	Loss: 0.0068(train)	|	Acc: 96.4%(train)
    	Loss: 0.0000(valid)	|	Acc: 89.6%(valid)
    Epoch: 4  | time in 0 minutes, 10 seconds
    	Loss: 0.0038(train)	|	Acc: 98.2%(train)
    	Loss: 0.0001(valid)	|	Acc: 90.5%(valid)
    Epoch: 5  | time in 0 minutes, 9 seconds
    	Loss: 0.0022(train)	|	Acc: 99.0%(train)
    	Loss: 0.0001(valid)	|	Acc: 90.8%(valid)

- 손실이 줄어들고, 정확도가 올라가는 것으로 보아 학습이 잘 되고 있음을 알 수 있다


### 테스트 데이터셋으로 모델 평가하기

```python
print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
```

    Checking the results of test dataset...
    	Loss: 0.0002(test)	|	Acc: 90.8%(test)

- 90 퍼센트의 정확도를 보인다


### 랜덤 뉴스에 테스트해보기

이때까지 제일 성능이 좋았던 모델을 사용해 랜덤 뉴스에 테스트 해본다

```python
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])
```

    This is a Sports news

- 성공