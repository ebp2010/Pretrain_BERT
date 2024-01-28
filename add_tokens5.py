# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:50:49 2022

@author: ichida

1)コーパスにあるデータを用いて、sentencepieceによるトークナイザーmodelを作成し、分かち書き、頻出順上位を抽出

py37TF4162@deep2
gptj@owner2@deep2

"""

from sentencepiece import SentencePieceTrainer
import sentencepiece as spm

drive_dir="//EBP-NAS01/edogawa/Keikakiroku/"
# drive_dir="D:/Okayama/"
# drive_dir="D:/BERT/Edogawa_BERT/keikakiroku/"

# 東北大モデルそのままのvocab=size 32000　のモデルパス
small_token_dir="D:/Edogawa/model_save/20231129 protection rehabilitation/protection/token/" 

# # vocab_sizeはエラーが出ない程度に大きいほうが熟語などが登録されるのでベター
SentencePieceTrainer.Train(
    '--input='+drive_dir+'corpus_2021-2023.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'edogawa_sp --character_coverage=0.9995 --vocab_size=20000 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
    # '--input='+drive_dir+'corpus/corpus.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=1201 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
)




from transformers import AlbertTokenizer,BertJapaneseTokenizer

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去される。 
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'edogawa_sp.model', keep_accents=True)
text = "児童相談所に本児は保護されたが、名前を言えない"
print(tokenizer.tokenize(text))

corpus = open(drive_dir+"corpus_2021-2023.txt", "r",encoding="utf-8_sig")
c = corpus.read()


# 以下は前処理で行っている場合は行わなくてもOK（時間を要する）
# import neologdn
# c = neologdn.normalize(c)

import unicodedata
c = unicodedata.normalize("NFKC", c)


c_t = tokenizer.tokenize(c)


words = c_t

import collections
import regex
import re

c_c = collections.Counter(words)
c_best = c_c.most_common(80000)

vocab_file = []
for i in c_best:
    ii=i[0]
    ii = re.sub(r"\d","",ii)
    # if len(ii)>1:
    rii = regex.findall("\p{Han}+", ii)
    if rii!=[] and len(ii)==len(rii[0]) and len(ii)>1:
        print(ii," ",rii)
        vocab_file.append(ii)
        
for i in c_best:
    ii=i[0]
    ii = re.sub(r"\d","",ii)
    # if len(ii)>1:
    rii = regex.findall("\p{Katakana}+", ii)
    if rii!=[] and len(ii)==len(rii[0]) and len(ii)>1:
        print(ii," ",rii)
        vocab_file.append(ii)    

for i in c_best:
    ii=i[0]
    ii = re.sub(r"\d","",ii)
    # if len(ii)>1:
    rii = regex.findall('[a-zA-Zａ-ｚＡ-Ｚ]+', ii)
    # rii = regex.findall('[Ａ-Ｚ]+', ii)
    if rii!=[] and len(ii)==len(rii[0]) and len(ii)>1:
        print(ii," ",rii)
        vocab_file.append(ii)   

#以下のミャンマーからの全角アルファベットを読み込むコードは多くを読み取りすぎてあまり使えず
# #Include Full-width alphabet like "ｃｗ" as well as "cw".
# FullWidth = unicodedata.normalize('NFKC', c)
# #print(FullWidth)

# #collect alphabetical words from corpus.txt file
# result = re.findall('[a-z]+', FullWidth, re.IGNORECASE)
# result_full = sorted(list(set(result)))
# print(result_full)    
 
# print(vocab_file)
# vocab_file = vocab_file + result_full
# print(vocab_file)




# eng = []
# for word in c.split():
#     word = re.findall('[A-Za-z]+', word)
#     if word:
#         eng.append(word[0])

# S = sorted(list(set(eng)))
# print(S)
# vocab_file = vocab_file + S
# print(vocab_file)

"""
２）既存の語彙ファイルvocab.txtに含まれない単語数をカウント
"""

CC_list = vocab_file
txt_file = open(small_token_dir + "/vocab.txt", "r",encoding="utf-8_sig")
# txt_file = open("D:/Edogawa/model_save/protection/token/vocab.txt", "r",encoding="utf-8_sig")
txt_A = txt_file.read()

A_list = txt_A.split("\n")

C_set = set(CC_list).difference(A_list)
#Type change, Set into list
C_list = list(C_set)

import pandas as pd
C_list_df = pd.DataFrame(C_list)

C_list_df.to_csv("D:/BERT/江戸川区向け辞書/経過記録固有の単語.csv")


"""
3)adding words to the tokenizer
"""


#語彙の追加
from transformers import AlbertTokenizer,BertJapaneseTokenizer

jtk_a = BertJapaneseTokenizer.from_pretrained(small_token_dir)
# jtk_a = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token/")

# jp_text = "休職期間満了まで後２ヶ月ほど。原職復帰が原則だが、本人の元いた部署は無くなっており、戻らせる場所が（地理的にも業務的にも）無く、会社としてはかなり困っている。"
jp_text = "内夫に叩かれた児は児童相談所に一時保護されたが、名前を言えない"

print("vocab size", len(jtk_a))
print(jtk_a.tokenize(jp_text))

jtk_a.add_tokens(C_list)
# jtk_a.add_tokens(["児童相談所"])
# jtk_a.add_tokens(["本児","児相","内夫"])


print("new vocab size", len(jtk_a))
print(jtk_a.tokenize(jp_text))


jtk_a.save_pretrained(drive_dir + '/tokenizer20240113_2/')






"""
vocab と　vocab.txtだけから追加する前処理コード
"""

# #1) Reading two text files (A & B)about vocabulary.
# # opening the vocab.txt　（追加先のファイルを開く）
# txt_file = open("D:/Edogawa/model_save/protection/token/vocab.txt", "r",encoding="utf-8_sig")
# # opening the vocab made from sentence piece　（追加したい語彙のファイルを開く）
# vocab_file = open("D:/BERT/token_add/vocab/B.vocab", "r",encoding="utf-8_sig")

# # reading the file
# txt_A = txt_file.read()
# vocab_B = vocab_file.read()

# #2) Making two list variables based on each text file.

# # replacing end splitting the text 
# # when newline ('\n') is seen.
# A_list = txt_A.split("\n")
# B_list = vocab_B.split("\n")

# #3) Making new list variable C that the same format of A

# C_list = [i.split('\t', 1)[0] for i in B_list]

# import regex

# CC_list=[]
# for w in C_list:
#     ww=regex.findall("\p{Han}+", w)    
#     if ww!=[] and len(ww)==1 and len(ww[0])>1:
#         print(ww)
#         CC_list.append(ww[0])
