# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:35:47 2021

以下のサイトのcmdによる手法のほうがよい場合あり
https://kimoto-sbd.co.jp/blog/2010/07/21806/

@author: owner
"""


import pandas as pd
import glob

folder_dir='C:/Users/owner/Desktop/BERT/keikakiroku/corpus0122/merge/'
# folder_dir=drive_dir+'corpus/kaigyozumi/corpus.txt'

# 読み込み
with open('sample.html') as reader:
    content = reader.read()
 
# 置換
content = content.replace('div>', 'p>')
 
# 書き出し
with open('sample_fixed.html', 'w') as writer:
    writer.write(content)
    


txt_files = glob.glob(folder_dir+'*.txt')
files = []
for f in txt_files:
    # files.append(pd.read_table(f, encoding="shift-jis"))
    files.append(pd.read_table(f, encoding='utf-8-sig')) 
df = pd.concat(files)
# print(df)

with open(folder_dir+'corpus_merged.txt', mode='w',encoding='utf-8-sig') as f:   
    f.writelines(df)