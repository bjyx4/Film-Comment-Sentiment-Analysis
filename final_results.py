# 获得模型在测试集上的评估结果

import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn import metrics

# 读取最佳模型
model = ClassificationModel("roberta", '/data/private/zjy/software/simpletransformers/outputs/best_model', num_labels=2)
# 读取二分类未分割测试集数据
test_df = pd.read_csv("data/test_bi_raw.csv",skip_blank_lines = True)
# 进行评估
preds, outputs = model.predict(test_df.text.tolist())
targets = test_df["labels"].tolist()
print("accuracy", metrics.accuracy_score(targets,preds))
print("f1 score", metrics.f1_score(targets,preds))
