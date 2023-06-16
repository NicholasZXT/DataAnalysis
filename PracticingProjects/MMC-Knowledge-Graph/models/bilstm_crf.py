import torch
from torch.nn import Module, LSTM, Linear, Dropout, Embedding
# from torchcrf import CRF  # 这是CRF实现的原版package
from pytorch_crf import CRF


class BiLstmCRF(Module):
    def __init__(self, num_tags, vocab_size, embedding_dim, lstm_size, lstm_layer_num, lstm_dropout):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.bilstm = LSTM(input_size=embedding_dim, hidden_size=lstm_size // 2, bidirectional=True,
                           num_layers=lstm_layer_num, batch_first=True, dropout=lstm_dropout)
        self.hidden2tag = Linear(in_features=lstm_size, out_features=num_tags)
        self.crf = CRF(batch_first=True, num_tags=num_tags)

    def forward(self, inputs):
        word_vec = self.embedding(inputs)
        bilstm_out = self.bilstm(word_vec)
        bilstm_emission = self.hidden2tag(bilstm_out)
        # 注意，forward 方法里，使用的是 decode，而不是通常的forward
        tags_pred = self.crf.decode(emissions=bilstm_emission)
        return tags_pred

    def forward_train(self, inputs, tags):
        word_vec = self.embedding(inputs)
        bilstm_out = self.bilstm(word_vec)
        bilstm_emission = self.hidden2tag(bilstm_out)
        # 这里才使用的是 CRF.forward
        loss = self.crf(emissions=bilstm_emission, tags=tags, reduction='sum')
        return loss


if __name__ == '__main__':
    batch_size, seq_length, num_tags = 2, 3, 5
    crf_model = CRF(num_tags=num_tags, batch_first=True)
    emissions = torch.randn(batch_size, seq_length, num_tags)
    tags = torch.tensor([[0, 2, 3], [1, 4, 1]], dtype=torch.long)
    # 这个得到的是每个 batch 的 真实路径的概率
    log_out = crf_model(emissions=emissions, tags=tags, reduction='none')
    # 这里做了 sum
    log_out_sum = crf_model(emissions=emissions, tags=tags, reduction='sum')

    # 使用维特比算法进行预测
    emissions_pred = torch.randn(1, seq_length, num_tags)
    # 得到序列中每个token的预测label序列
    tags_pred = crf_model.decode(emissions_pred)
