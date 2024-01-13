# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:35:47 2021

以下のサイトのcmdによる手法のほうがよい場合あり
https://kimoto-sbd.co.jp/blog/2010/07/21806/

@author: owner
"""


import pandas as pd
import glob

folder_dir='C:/Users/owner/Desktop/BERT/corpus/kaigyozumi/'
# folder_dir=drive_dir+'corpus/kaigyozumi/corpus.txt'

# 読み込み
with open(folder_dir+'corpus.txt', encoding='utf-8-sig') as reader:
    content = reader.read()
 
# 置換
content = content.replace('[', '')
content = content.replace(']', '')
content = content.replace("'","")
content = content.replace("\t","")

 
# 書き出し
with open(folder_dir+'corpus+.txt', 'w', encoding='utf-8-sig') as writer:
    writer.write(content)

