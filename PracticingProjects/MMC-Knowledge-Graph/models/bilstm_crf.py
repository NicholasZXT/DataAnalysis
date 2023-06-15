import torch
from torch.nn import Module, LSTM, Linear, Dropout, Embedding
from torchcrf import CRF


class BiLstmCRF(Module):
    def __init__(self, num_tags, embedding_dim, lstm_size, lstm_layer_num, lstm_dropout):
        super().__init__()
        self.embedding = Embedding(num_embeddings=512, embedding_dim=embedding_dim, padding_idx=0)
        self.bilstm = LSTM(input_size=embedding_dim, hidden_size=lstm_size, num_layers=lstm_layer_num, batch_first=True,
                           dropout=lstm_dropout, bidirectional=True)
        self.linear = Linear(in_features=lstm_size, out_features=num_tags)
        self.crf = CRF(batch_first=True, num_tags=num_tags)

    def forward(self, inputs):
        word_vec = self.embedding(inputs)
        bilstm_out = self.bilstm(word_vec)
        bilstm_out_linear = self.linear(bilstm_out)
        pass


if __name__ == '__main__':
    batch_size, seq_length, num_tags = 2, 3, 5
    crf_model = CRF(num_tags=num_tags, batch_first=True)
    emissions = torch.randn(batch_size, seq_length, num_tags)
    tags = torch.tensor([[0, 2, 3], [1, 4, 1]], dtype=torch.long)
    log_out = crf_model(emissions=emissions, tags=tags, reduction='none')
    log_out_sum = crf_model(emissions=emissions, tags=tags, reduction='sum')