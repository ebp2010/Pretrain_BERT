# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:00:15 2021

@author: ichida
"""
import pandas as pd
import re
from merging_csv_folder import merging_csv
from maeshori_deleting_wrong_letters import data_cleaning
import glob
import codecs as cd

folder_dir='D:/Okayama/corpus/'
# folder_dir='//EBP-NAS01/edogawa/Keikakiroku/Keikakiroku_June2022/'
# folder_dir='D:/Edogawa/Edogawa_BERT/keikakiroku/original12files/'

def read_csv_as_str(file_path):
    with cd.open(file_path , "r", "utf-8", "ignore") as csv_file: # with cd.open(direc+"愛媛2021_回答_小林集計.csv", "r", "utf-8", "ignore") as csv_file: 
        df_csv = pd.read_csv(csv_file) 
    # df_csv = pd.read_csv(file_path, dtype=str, header=0,index_col=None)
    return df_csv

def merging_many_csv(folder_path):
    file_paths = glob.glob(folder_path + '*.csv')
    
    files =[]
    for path in file_paths:
        # file = pd.read_csv(path, dtype=str, header=0,index_col=None)
        file = read_csv_as_str(path)
        files.append(file)
        
    df_merged = pd.concat(files, axis=0)
    df = df_merged.reset_index()
    return df

#フォルダ内のCSVを読み込みすべて結合 txtでも読み込み可能
df1=merging_many_csv(folder_dir)
# df1=pd.read_csv(folder_dir+'original12files/01.csv')


df1=df1.loc[:,'message']
# print(df2.head())
# print(df2.shape)

#空白のセルを削除しリスト化
df22=df1.dropna()
df3=df22.values.tolist()

result = []
df4 = []
for i in df3:
    space_modified = re.sub(r"[\u3000 \t　\s　\r\n]", "", i)
    df4.append(space_modified)

for s in df4:
  # s ='[CLS]' +  s 
  s = s.replace("[", "")
  s = s.replace("]", "")
  x = s.replace("。", "。\n")
  xx = x + "\n"
  # x = re.sub("・", "", kaigyo)
  result.append(xx)
  

with open(folder_dir+'corpus.txt', mode='w',encoding='utf-8-sig') as f:   
    f.writelines(result)
    

# result2=data_cleaning(result)

# with open(folder_dir+'corpus_cleaned.txt', mode='w',encoding='utf-8-sig') as f:   
#     f.writelines(result)
