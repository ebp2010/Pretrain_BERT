# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:00:15 2021

@author: ichida

gptj@owner2@deep2
"""
import pandas as pd
import numpy as np
import re
# from merging_csv_folder import merging_csv
# from maeshori_deleting_wrong_letters import data_cleaning
import glob
import codecs as cd
import unicodedata

folder_paths=[
    # '//EBP-NAS01/edogawa/Keikakiroku/2021_8_Keikakiroku_Aug/',
    #           '//EBP-NAS01/edogawa/Keikakiroku/2022_6_Keikakiroku_June/',#形式が異なる
              '//EBP-NAS01/edogawa/Keikakiroku/2022_10_Keikakiroku_Oct/',
             '//EBP-NAS01/edogawa/Keikakiroku/2023_8_Keikakiroku_Aug/'
             ]
# folder_dir='D:/Okayama/corpus/'
# folder_dir='//EBP-NAS01/edogawa/Keikakiroku/Keikakiroku_June2022/'
# folder_dir='D:/Edogawa/Edogawa_BERT/keikakiroku/original12files/'


def read_csv_as_str(file_path):
    with cd.open(file_path , "r", "utf-8", "ignore") as csv_file: # or "utf-8" "shift-jis"  
        df_csv = pd.read_csv(csv_file) 
    # df_csv = pd.read_csv(file_path, dtype=str, header=0,index_col=None)
    return df_csv

def merging_many_csv(folder_paths):
    file_paths_ap=[]
    for folder_path in folder_paths:       
        file_paths = glob.glob(folder_path + '*.csv')
        file_paths_ap = file_paths_ap + file_paths      
            
    files =[]
    for path in file_paths_ap:
        # file = pd.read_csv(path, dtype=str, header=0,index_col=None)
        file = read_csv_as_str(path)
        print(file.head())
        file = file.loc[:,["ケースID","内容","登録日時","主訴"]]
        files.append(file)
        
    df_merged = pd.concat(files, axis=0)
    df = df_merged.reset_index()
    return df

#フォルダ内のCSVを読み込みすべて結合 txtでも読み込み可能
df0 = merging_many_csv(folder_paths)
# df1=pd.read_csv(folder_dir+'original12files/01.csv')

df1 = df0.sort_values(["ケースID","登録日時"])

df1_s = df1["主訴"].replace(np.nan,"")
df1_s = df1_s.drop_duplicates()

df1 = df1.drop(columns=["index","登録日時","主訴"])
df1 = df1.drop_duplicates()
df1 = df1.dropna()

df11 = pd.concat([df1,df1_s], axis=1)

grouped = df11.groupby("ケースID")
df12 = grouped.sum()

df12["主訴"] = df12["主訴"].replace(0,"")
df12["内容"] = "\n" + "\n" + df12["主訴"] + df12["内容"]

df12 = df12.drop(columns=["主訴"])

# df1=df1.loc[:,'内容']
# print(df2.head())
# print(df2.shape)

#リスト化
# df3=df12
# df3=df12.values
df3=df12["内容"].values.tolist()

result = []
df4 = []
for i in df3:
    space_modified = re.sub(r"[\u3000 \t　\s　\r\n]", "", i)
    df4.append(space_modified)

for s in df4:
  # s ='[CLS]' +  s 
  s = s.replace("[", "")
  s = s.replace("]", "") 
  s = s.replace("-", "")
  s = unicodedata.normalize("NFKC",s)
  x = s.replace("。", "。\n")
  xx = x + "\n"
  # x = re.sub("・", "", kaigyo)
  result.append(xx)
  
  
# 20240127追加　（本ファイル内では未テスト）  
# import neologdn
# result = neologdn.normalize(result)

# result = unicodedata.normalize("NFKC", result)


with open('corpus.txt', mode='w',encoding='utf-8-sig') as f:   
    f.writelines(result)
    

# result2=data_cleaning(result)

# with open(folder_dir+'corpus_cleaned.txt', mode='w',encoding='utf-8-sig') as f:   
#     f.writelines(result)
