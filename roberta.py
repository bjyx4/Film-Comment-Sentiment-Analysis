# 使用不同的数据在roberta上进行实验，包括：
# 二分类未分割数据；二分类分割数据，
# 多分类未分割数据，多分类分割数据

import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score,f1_score


model_args = {
    'output_dir': 'roberta-roberta-large-outputs',
    'max_seq_length': 512,
    'num_train_epochs': 3,
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'gradient_accumulation_steps': 1,
    'learning_rate': 4e-5,
    'save_steps': 3000,
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


# 二分类模型的训练和评估
def binary(train,eval):
    train_df = pd.read_csv(train, skip_blank_lines=True)
    eval_df = pd.read_csv(eval, skip_blank_lines=True)
    # Create a ClassificationModel
    model = ClassificationModel("roberta", "roberta-large", num_labels=2, args=model_args)
    # Train the model
    model.train_model(train_df,eval_df)
    # evaluate the model
    preds, outputs = model.predict(eval_df.text.tolist())
    targets = eval_df.labels.tolist()
    print("accuracy", accuracy_score(targets, preds))
    print("f1 score", f1_score(targets, preds))

# 多分类模型的训练和评估
def multi_class(train,eval):
    train_df = pd.read_csv(train, skip_blank_lines=True)
    eval_df = pd.read_csv(eval, skip_blank_lines=True)
    # Create a ClassificationModel
    model = ClassificationModel("roberta", "roberta-large", num_labels=8, args=model_args)
    # Train the model
    model.train_model(train_df,eval_df)
    # evaluate the model
    preds, outputs = model.predict(eval_df.text.tolist())
    bi_preds = multi2binary(preds)
    targets = eval_df.labels.tolist()
    bi_targets = multi2binary(targets)
    print("accuracy", accuracy_score(bi_targets, bi_preds))
    print("f1 score", f1_score(bi_targets, bi_preds))


# 将多分类标签转化成二分类标签
def multi2binary(labels):
    bi_labels = []
    for label in labels:
        if label < 4:
            bi_labels.append(0)
        else:
            bi_labels.append(1)
    return bi_labels

# 测试二分类未分割数据；
#binary("data/train_bi_raw.csv","data/validation_bi_raw.csv")

# 测试二分类分割数据
binary("data/train_bi_split.csv","data/validation_bi_split.csv")

# 测试多分类未分割数据
#multi_class("data/train_multi_raw.csv","data/validation_multi_raw.csv")

# 测试多分类分割数据
#multi_class("data/train_multi_split.csv","data/validation_multi_split.csv")



