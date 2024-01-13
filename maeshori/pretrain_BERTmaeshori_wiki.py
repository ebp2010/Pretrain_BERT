# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 21:00:15 2021

@author: ichida tochuu Deep1 ni kanseiban ari

"""

import re

drive_dir='C:/Users/owner/Desktop/BERT/'
path=drive_dir+'corpus/wiki_wakati_all.txt'

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    s = f.read()
    print(type(s))
    # print(s)

# s2 = re.sub(r"[\u3000 \t　\s　\r\n]", "", s)


s = re.sub("。 ","。", s) 
s = re.sub("\'","", s) 
s = re.sub("\]","", s) 
s = re.sub("\[","", s)                          
s = re.sub(r"[\u3000 \t　\s]", r"\t", s) # 空白で区切られたファイルをタブ区切りへ

kaigyo = s.replace("。", "。\n") 
x = re.sub("・", "", kaigyo)

with open(drive_dir+'corpus/kaigyozumi/corpus.txt', mode='w',encoding='utf-8-sig') as f:   
    f.writelines(x)





# df1=pd.read_csv(drive_dir+'corpus/corpus.csv')
# df2=df1.loc[:,'Text']
# print(df2.head())
# # df2=df2.transpose()
# # print(df2)

# df3=df2.values.tolist()

# result = []
# import re
# df4 = []
# for i in df3:
#   space_modified = re.sub(r"[\u3000 \t　\s　\r\n]", "", i)
#   df4.append(space_modified)
# print(df4)

# for s in df4:
#   # s ='[CLS]' +  s 
#   kaigyo = s.replace("。", "。\n") 
#   x = re.sub("・", "", kaigyo)
#   result.append(x)
# with open(drive_dir+'corpus/corpus.txt', mode='w',encoding='utf-8-sig') as f:   
#     f.writelines(result)

# import textwrap
  # x12 = textwrap.wrap(kaigyo, 100)
  # x13='\n'.join(x12)


# with open(drive_dir+'corpus/corpus.txt', encoding="sjis") as fd:
#     for line in fd:
#         line = line.rstrip()
#         print(line)



#     for line in s:
#         line = line.rstrip()
#         print(line)
# text = """
# ・まめの木式ペアレント・トレーニングをベースとした、短縮版のプログラムを実施。 約半年間計６回のセッションを、児童心理司１名、児童福祉司１名で実施。
# """
# #text_x = re.sub(extraction, ''.ljust(1, 'x'), text)


# extraction = re.search("。 ", text).group()
# text_x = text.replace(extraction,"。"+r'\n'.ljust(1, 'x'))
# text_x='[CLS]'+text_x

# print(text_x)


# # 冒頭に[CLS]を挿入し、「。」の後に改めて改行コードを挿入
# result = []
# for s in df4:
  # x1 = re.sub(" ", "", i)
  # # # 改行コード
  # # x = re.sub("\n", "", x)
  # # 全角スペース
  # x2 = re.sub("　", "", x1)
  # # 全角スペース
  # x31 = re.sub("。","。"+r'\n', x2)
  # x3 = re.sub("。 ・","。"+r'\n', x31)
  # x4='[CLS]'+x3
  # extraction = re.search("。", x).group()
  # x = x.replace(extraction,"。"+r'\n'.ljust(1, 'x'))
  # x = x.replace("。", "\n。\n")
  # x = x.strip().replace("\n\n", "\n")
  # x10 = re.sub("＜","＜"+r'\n', x4)
  # x11 = re.sub("＞","＞"+r'\n', x10)
  # # 特殊文言
  # x = re.sub('\u2666', "", x)
  # # 特殊文言
  # x = re.sub('\u2022', "", x)
  # # 特殊文言
  # x = re.sub('\U00020b9f', "", x)
  # print( x )


# tokenized_text.insert(0, '[CLS]')
#  tokenized_text.insert(11, '[SEP]') # 複文対応
#  tokenized_text.append('[SEP]')
#  masked_index = 14
#  tokenized_text[masked_index] = '[MASK]'
#  print(tokenized_text)


# sample_list =df3
# result = []
# import re
# import textwrap
# # for文を使用して配列の要素を一つずつ取り出す
# for i in sample_list:
#   # for文の中で配列から要素が一つずつ変数'i'に取り出される
#   # 数字を0に変換
#   # cleaned_text = re.sub(r'\d+', '0', i)
#   # 半角スペース
#   cleaned_text = re.sub(" ", "", i)
#   # # 改行コード
#   # cleaned_text = re.sub("\n", "", cleaned_text)
#   # 全角スペース
#   cleaned_text = re.sub("　", "", cleaned_text)
#   # 全角スペース
#   cleaned_text = re.sub("。","。"+r'\n', cleaned_text)
  
#   # extraction = re.search("。", cleaned_text).group()
#   # cleaned_text = cleaned_text.replace(extraction,"。"+r'\n'.ljust(1, 'x'))
  
#   # cleaned_text = cleaned_text.replace("。", "\n。\n")
#   # cleaned_text = cleaned_text.strip().replace("\n\n", "\n")
#   # 全角スペース
#   cleaned_text = re.sub("＜","＜"+r'\n', cleaned_text)
#   # 全角スペース
#   cleaned_text = re.sub("＞","＞"+r'\n', cleaned_text)
  
#   cleaned_text = textwrap.wrap(cleaned_text, 100)
#   cleaned_text='\n'.join(cleaned_text)
  
#   # # 特殊文言
#   # cleaned_text = re.sub('\u2666', "", cleaned_text)
#   # # 特殊文言
#   # cleaned_text = re.sub('\u2022', "", cleaned_text)
#   # # 特殊文言
#   # cleaned_text = re.sub('\U00020b9f', "", cleaned_text)
#   # print( cleaned_text )
#   result.append(cleaned_text)
# print(result)
# with open(drive_dir+'corpus/corpus.txt', mode='w',encoding='utf-8-sig') as f:   
#     f.writelines(result)

  

# df3=df2.replace({'保護者':'親','TX': 'Texas'})
# print(df3.head())

# s_copy = df2.copy()
# s_copy.replace({'者':'親'}, inplace=True)
# print(s_copy)