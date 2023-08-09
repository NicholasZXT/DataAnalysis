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
    pass
