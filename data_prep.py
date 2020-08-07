"""对两个数据集进行初步处理：
第一步：获取每个样例的rating值和文本；
第二步：将rating值处理为0-7的label；
第三步：剔除文本中<br \>；
第四步：使用NLTK对文本进行tokenize和Lemmatize (词形还原)；
第五步：从训练集中分出15%的数据作为验证集。
第六步：将处理好的训练集、验证集、测试集数据分别输出到csv文件中，并储存在\data目录下，以便后续使用。"""

import pandas as pd
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


def data_prep(data):
    df = pd.DataFrame(columns=("labels","text"))
    index = 0
    wordnet_lemmatizer = WordNetLemmatizer()
    for file in Path(data).rglob("*.txt"):
        # 将rating值处理为0-7的label
        if file.parent.stem == "neg":
            label = int(file.stem.split("_")[1]) - 1
        else:
            label = int(file.stem.split("_")[1]) - 3
        with open(file,encoding="utf-8") as f:
            text = f.read().strip()
            #去除<br />符号
            text = re.sub("(\<br /\>)+"," ",text)
            # 对文本进行tokenize
            sents = sent_tokenize(text)
            tokens = []
            for sent in sents:
                tokens += word_tokenize(sent)
            # 词形还原
            lemma_tokens = []
            for token in tokens:
                word1 = wordnet_lemmatizer.lemmatize(token, pos="n")
                word2 = wordnet_lemmatizer.lemmatize(word1, pos="v")
                word3 = wordnet_lemmatizer.lemmatize(word2, pos=("a"))
                lemma_tokens.append(word3)
            # 将label和文本存入dataframe
            content = " ".join(lemma_tokens)
            df.loc[index] = [label,content]
            index += 1
    return df


#数据初步处理
df_train = data_prep("dataset/train")
df_test = data_prep("dataset/test")

#从训练集中划分出15%数据作为验证集
df_train,df_val = train_test_split(df_train,test_size=0.15,stratify=df_train['labels'],random_state=1)

# 统计训练集的文本长度并画图
df_train.text.apply(lambda x:len(x.split(" "))).plot(kind = "hist")
plt.show()

# 将处理好的训练、测试、验证数据分别保存到csv文件中
df_train.to_csv("data/train_multi_raw.csv",index=False)
df_val.to_csv("data/validation_multi_raw.csv",index=False)
df_test.to_csv("data/test_multi_raw.csv",index=False)





