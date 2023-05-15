# Open-BERT

You can download individually from the table below:

|   |H=128|
|---|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**][2_128]|

I have evaluated the BERT-Tiny a second time to get a fair comparison.

| Model            | Score | CoLA  | SST-2 |   MRPC    |   STS-B   |    QQP    | MNLI-m | MNLI-mm | QNLI(v2) |  RTE  | WNLI  |
|------------------|:-----:|:-----:|:-----:|:---------:|:---------:|:---------:|:------:|:-------:|:--------:|:-----:|:-----:|
| BERT-Tiny-origin | 64.2  |  0.0  | 83.2  | 81.1/71.1 | 74.3/73.6 | 62.2/83.4 |  70.2  |  70.3   |   81.5   | 57.2  | 62.3  |
| BERT-Tiny  | 65.1  | 69.12 | 79.12 |   70.34   |   42.73   |   79.81   | 64.60  |  66.48  |  77.63   | 59.21 | 42.25 |

[origin]: https://github.com/google-research/bert
[evaluation]: colab%2Fbert_evaluation_on_bert-tiny.ipynb
[2_128]: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip

This example code fine-tunes `BERT-Base` on the TASKs, which contains thousands examples and can fine-tune in a
few minutes on a T4 GPU. An accuracy score is displayed after each execution and is recorded for model comparison.

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=TASK \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/TASK \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --output_dir=/content/trained_output
```

The evaluation is completed on Google Colab with a T4 GPU, with following software:

* Python 3.6.15
* Ubuntu 18.04
* Tensorflow-gpu 1.13.1