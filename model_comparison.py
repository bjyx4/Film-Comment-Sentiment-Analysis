# 比较三个预训练模型在多分类分割数据上的表现：
# bert-large-cased;
# xlnet-large-cased;
# roberta-large

from sys import argv
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn import metrics

train_df = pd.read_csv("data/train.csv",skip_blank_lines = True)
eval_df = pd.read_csv("data/validation.csv",skip_blank_lines = True)
model_type = argv[1]
model_name = argv[2]

model_args = {
    'output_dir': f'{model_type}-{model_name}-outputs',
    'max_seq_length': 350,
    'num_train_epochs': 3,
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'gradient_accumulation_steps': 1,
    'learning_rate': 4e-5,
    'save_steps': 3000, #每3000步保存一个checkpoint
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 3000,
    'reprocess_input_data': True,
    "save_model_every_epoch": True,
    'overwrite_output_dir': True,
    'no_cache': True,
    'use_early_stopping': True,
    'early_stopping_patience': 3,
    'manual_seed': 1,
    'n_gpu': 8
}


print(model_type)
print(model_name)

# Create a ClassificationModel
model = ClassificationModel(model_type, model_name, num_labels=8, args=model_args)

# Train the model
model.train_model(train_df,eval_df)

# multi-class: Evaluate the model
def multi2binary(labels):
    bi_labels = []
    for label in labels:
        if label < 4:
            bi_labels.append(0)
        else:
            bi_labels.append(1)
    return bi_labels

preds, outputs = model.predict(eval_df.text.tolist())
bi_preds = multi2binary(preds)
targets = eval_df.labels.tolist()
bi_targets = multi2binary(targets)
print("accuracy", metrics.accuracy_score(bi_targets,bi_preds))
print("f1 score", metrics.f1_score(bi_targets,bi_preds))


