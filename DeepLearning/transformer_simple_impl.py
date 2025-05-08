"""
来自哈佛NLP团队的 The Annotated Transformer (http://nlp.seas.harvard.edu/annotated-transformer)
"""
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
# import altair as alt
# from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
# from torchtext.vocab import build_vocab_from_iterator
# import torchtext.datasets as datasets
import spacy
# import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DD

# ----------------------------------------------------------------------
# 一些辅助函数
# Some convenience helper functions used throughout the notebook
RUN_EXAMPLES = True
def is_interactive_notebook():
    return __name__ == "__main__"

def show_example(fn, args=None):
    if args is None:
        args = []
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=None):
    if args is None:
        args = []
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None

# ------------------------------------------------------------------------------------
# 构建一个 Transformer 模型，原博客按照 自顶向下 的方式介绍引入的

class EncoderDecoder(nn.Module):
    """A standard Encoder-Decoder architecture. Base for this and many other models."""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder  # 下面的 Encoder 对象
        self.decoder = decoder  # 下面的 Decoder 对象
        self.src_embed = src_embed  # 输入序列的 embedding 层
        self.tgt_embed = tgt_embed  # 输出序列的 embedding 层
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # encoder 的输入只需要输入序列 src 和对应的 mask即可
        # 返回的shape=(batch_size, seq_len, d_model)
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memory 是 self.encode() 的输出，也就是 Encoder 的输出 --------------------------- KEY
        # tgt 才是 目标输出的 embedding 序列
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """Construct a LayerNorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # MultiHeadedAttention 对象
        self.feed_forward = feed_forward  # PositionwiseFeedForward 对象
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        # 输入只有一个 x
        # 调用 MultiHeadedAttention 时，query/key/value 都是 x ------------------------- KEY
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 输出shape= (batch_size, seq_len, d_model)
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Decoder 的输入不止 x，还有 memory
        # x 是目标序列的 embedding 序列，memory 是 encoder 的输出序列
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # decode层里有两个 MultiHeadedAttention 对象 -------------------------------------- KEY
        self.self_attn = self_attn  # decoder：self-attention
        self.src_attn = src_attn    # decoder：encoder-decoder attention
        self.feed_forward = feed_forward  # PositionwiseFeedForward 对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x 是目标序列的 embedding 序列，memory 是 encoder 的输出序列
        """Follow Figure 1 (right) for connections."""
        m = memory
        # self-attention 的输入的 query/key/value 都是 x，也就是上一层的输入
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # encoder-decoder attention 的 query 来自上一层，但是 key/value 都是使用的 memory，它实际上是 encoder 的输出 ------ KEY
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask_res = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask_res == 0

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # 计算单个head 的 attention: Attention(Q,K,V) = softmax(Q * K' / sqrt(d_k)) * V
    # query/key/value 的shape是: (batch_size, h, seq_len, d_k)，计算attention时，只操作最后两个维度
    d_k = query.size(-1)
    # 计算score，shape变化如下
    # (batch_size, h, seq_len, d_k) * (batch_size, h, d_k, seq_len) = (batch_size, h, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 有 mask 的话，对要 mask 的位置(0) 填充一个极小的值
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 计算 softmax 后的值
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 乘以 V，shape变化为：(batch_size, h, seq_len, seq_len) * (batch_size, h, seq_len, d_k) = (batch_size, h, seq_len, d_k)
    # 还返回了 score
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        # h 是 head 个数，d_model 是隐藏层维度
        # 第 i 个head下：WQ_i 为 (d_model, d_k), QK_i 为 (d_model, d_k)，QV_i 为 (dv, d_model)
        # We assume d_v always equals d_k
        assert d_model % h == 0
        self.d_k = d_model // h  # d_k 是每个 head 的维度
        self.h = h
        # 4个线性变换层也就是4个矩阵，Linear第一个参数对应的是 query/key/value 的embedding维度——也就是 d_model，第二个参数对应于 d_model
        # 注意，这里的4个矩阵都是同时对所有 head 进行操作，因为本来每个 head 对应的 矩阵维度应该是 Linear(d_model, d_model/h)，这里拼到一起了
        # 前3个分别用于 query, key, value，最后一个用于合并所有head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 保存 attention 的中间结果，即 softmax(score) 这个值
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        # query, key, value 的 shape 为：(batch_size, seq_len, embedding_dim)，其中 embedding_dim = d_model
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # query 的第一个维度是 batch_size
        nbatches = query.size(0)

        # MHA具体计算
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            # 每个元祖对进行如下计算：
            # 1. lin(x): query/key/value * linear(d_model, d_model) -> 输出shape: [batch_size, seq_len, d_model]
            #    这一步其实是同时计算了 h 个 head，每个head维度为 d_k, 而 d_k * h = d_model
            # 2. view(): shape变为 (batch_size, seq_len, h, d_k)，也就是将最后的 d_model 拆成了 h * d_k，把每个head的d_k拆出来了
            # 3. transpose(1,2): shape 变为 (batch_size, h, seq_len, d_k)，这是为了后续进行每个 head 内的 attention，因为每个head
            #    内的attention计算，用到的只有最后面的两个维度 (seq_len, d_k)
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            # 形成 3 个元祖对：(linear, query), (linear, key), (linear, value)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # 计算每个 head 内 Q,K,V 的 attention，输入和输出的 shape 都是 (batch_size, h, seq_len, d_k)
        # attn 是 softmax(score)，shape为 (batch_size, h, seq_len, seq_len)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # 将多个 head 的结果合并回去，shape = (batch_size, seq_len, d_model)
        x = (
            # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k)
            x.transpose(1, 2)
            .contiguous()  # 保证内存存储连续性
            # (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, h*d_k) -> (batch_size, seq_len, d_model)
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        # 再对最后的结果做一次变化，shape = (batch_size, seq_len, d_model)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, x * W1 + b1) * W2 + b2 = Relu(x * W1 + b1) * W2 + b2
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # vocab 是词典大小，d_model 是嵌入词向量的维度
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 这里对embedding的结果乘以了一个缩放系数
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * - (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 上面生成的 positional encoding 参数是固定的，不随训练过程变化，因此注册到 buffer 里
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 输入的 x 是 embedding 向量，直接和 PE 相加，并且 PE 向量不需要进行梯度更新
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)

def run_tests():
    for _ in range(10):
        inference_test()

show_example(run_tests)


# ------------------------------------------------------------------------------------
# 模型训练
class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

def rate(step, model_size, factor, warmup):
    """we have to default the step to 1 for LambdaLR function to avoid zero raising to negative power."""
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data
