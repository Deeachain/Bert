export BERT_BASE_DIR=/media/ding/Data/model_weight/nlp_weights/tensorflow/google/multilingual_L-12_H-768_A-12
export GLUE_DIR=/media/ding/Storage/competition/kaggle/jigsaw/data

python run_classifier.py \
  --task_name=jigsaw \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_lower_case=true \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=5 \
  --output_dir=./checkpoint/jigsaw/