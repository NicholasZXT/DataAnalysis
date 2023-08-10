import torch
from torch.nn import Module, LSTM, Linear, Dropout, Embedding
from torch import optim
from torch.utils.data import Dataset, DataLoader, default_collate

# from torchcrf import CRF  # 这是CRF实现的原版package
from pytorch_crf import CRF

from data_script import ENTITY_TYPES, MmcVocabulary, EntityTagMap, MmcDatasetV1CollateFun, MmcDatasetV1
from data_script import DATA_PATH, DATA_PROC_PATH


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
        bilstm_out, _ = self.bilstm(word_vec)
        bilstm_emission = self.hidden2tag(bilstm_out)
        # 注意，forward 方法里，使用的是 decode，而不是通常的forward
        tags_pred = self.crf.decode(emissions=bilstm_emission)
        return tags_pred

    def forward_train(self, inputs, tags):
        word_vec = self.embedding(inputs)
        bilstm_out, _ = self.bilstm(word_vec)
        bilstm_emission = self.hidden2tag(bilstm_out)
        # 这里才使用的是 CRF.forward
        loss = self.crf(emissions=bilstm_emission, tags=tags, reduction='sum')
        return -loss


if __name__ == '__main__':
    vocab = MmcVocabulary.generate_vocabulary(DATA_PATH)
    entity_map = EntityTagMap.generate_entities_bioes_tag(ENTITY_TYPES)
    # mmc_ds = MmcDatasetV1(data_dir=DATA_PROC_PATH, vocab=vocab, entity_types=ENTITY_TYPES, debug_limit=20)
    mmc_ds = MmcDatasetV1(data_dir=DATA_PROC_PATH, vocab=vocab, entity_types=ENTITY_TYPES)
    collate_fn = MmcDatasetV1CollateFun(max_len=128, entity_map=entity_map, vocab=vocab)
    mmc_loader = DataLoader(dataset=mmc_ds, batch_size=128, collate_fn=collate_fn)

    embed_size = 128
    lstm_size = 128
    lstm_dropout = 0.1
    lstm_layer_num = 2
    bilstm_crf = BiLstmCRF(num_tags=entity_map.tag_num, vocab_size=vocab.vocab_size, embedding_dim=embed_size,
                           lstm_size=lstm_size, lstm_layer_num=lstm_layer_num, lstm_dropout=lstm_dropout)
    bilstm_crf.cuda()
    optimizer = optim.SGD(params=bilstm_crf.parameters(), lr=0.1)

    epoch_num = 5
    for epoch in range(1, epoch_num+1):
        print(f"********** running train for epoch: {epoch} **********")
        for batch, (sent_idx_tensor, sent_tags, sent_tags_idx_tensor) in enumerate(mmc_loader, start=1):
            optimizer.zero_grad()
            # loss = bilstm_crf.forward_train(sent_idx_tensor, sent_tags_idx_tensor)
            loss = bilstm_crf.forward_train(sent_idx_tensor.cuda(), sent_tags_idx_tensor.cuda())
            print(f"training loss in epoch [{epoch}] at batch [{batch}] is: {loss}")
            loss.backward()
            optimizer.step()
