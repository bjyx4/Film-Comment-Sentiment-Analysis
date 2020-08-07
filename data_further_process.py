# 对数据进行进一步处理

import pandas as pd

#将过长的文本分割成短文本(有重叠的分割)
def split_text(text, split_len, overlap_len):
    text_piece=[]
    tokens = text.split(" ")
    num = len(tokens)//split_len
    window = split_len - overlap_len
    for i in range(num):
        if i == 0: #第一次分割,直接按分割长度取文本
            piece = tokens[:split_len]
        else: # 否则, 往回退overlap长度后继续往后按分割长度取文本
            piece = tokens[(i * window ):(i * window + split_len)]
        text_piece.append(piece)
    last_piece = tokens[(num * window - 1):]
    if last_piece:
        text_piece.append(last_piece)
    return text_piece

def get_split_data(path,type,class_num):
    df = pd.read_csv(path,skip_blank_lines=True)
    for index,row in df.iterrows():
        if len(row["text"].split(" ")) > 300:
            pieces = split_text(row["text"], 300, 50)
            label = row["labels"]
            for piece in pieces:
                insert_row = pd.DataFrame([[label," ".join(piece)]],columns=["labels","text"])
                above = df.loc[:index-1]
                below = df.loc[index+1:]
                df = above.append(insert_row, ignore_index=True).append(below, ignore_index=True)
                index += 1
    df.to_csv(f"data/{type}_{class_num}_split.csv",index=None)
    return df


# 将多分类数据处理成二分类数据
def get_binary_data(path,type,mode):
    df = pd.read_csv(path,skip_blank_lines = True)
    for index,row in df.iterrows():
        if row["labels"] > 3:
            df.at[index,"labels"] = 1
        else:
            df.at[index, "labels"] = 0
    df.to_csv(f"data/{type}_bi_{mode}.csv",index=None)
    return df


# 将多分类未分割数据转化成多分类分割数据
def main_split():
    df_train,split_locs_train = get_split_data("data/train_multi_raw.csv","train","multi")
    df_test,split_locs_test = get_split_data("data/test_multi_raw.csv","test","multi")
    df_validation = get_split_data("data/validation_multi_raw.csv","validation","multi")

def main_bi():
    #将多分类未分割数据转化成二分类未分割数据
    get_binary_data("data/train_multi_raw.csv","train","raw")
    get_binary_data("data/test_multi_raw.csv","test","raw")
    get_binary_data("data/validation_multi_raw.csv","validation","raw")

    #将多分类分割数据转化成二分类分割数据
    get_binary_data("data/train_multi_split.csv","train","split")
    get_binary_data("data/test_multi_split.csv","test","split")
    get_binary_data("data/validation_multi_split.csv","validation","split")


main_split()
#main_bi()