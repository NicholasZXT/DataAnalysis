# 用于练习 huggingface 的 transformer


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sentence = "A Titan RTX has 24GB of VRAM"
tokenized_sentence = tokenizer.tokenize(sentence)
print(tokenized_sentence)

encode_sentence_tokens = tokenizer(sentence)
print(encode_sentence_tokens)

decode_sentence = tokenizer.decode(encode_sentence_tokens['input_ids'])
print(decode_sentence)