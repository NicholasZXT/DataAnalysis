
# 以下是MRPC示例的运行参数配置
# 官方原版
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

--task_name=MRPC \
--do_train=true \
--do_eval=true \
# 使用cased的模型时，需要设置如下这个参数
--do_lower_case=False \
--data_dir=$GLUE_DIR/MRPC \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=2.0 \
--output_dir=./mrpc_output/


# 在windows上运行使用如下配置，特别是路径名的配置，windows下不能使用环境变量名展开路径
--task_name=MRPC \
--do_train=true \
--do_eval=true \
# 使用cased的模型时，需要设置如下这个参数
--do_lower_case=False \
--data_dir=./glue_data/MRPC \
--vocab_file=./bert-pre-trained-models\cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=./bert-pre-trained-models\cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=./bert-pre-trained-models\cased_L-12_H-768_A-12/bert_model.ckpt \
# 下面这两个，单机情况下，很容易OOM，需要改小一点
--max_seq_length=64 \
--train_batch_size=16 \
--learning_rate=2e-5 \
--num_train_epochs=2.0 \
--output_dir=./output/mrpc_output/