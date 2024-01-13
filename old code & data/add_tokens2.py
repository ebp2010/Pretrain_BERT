# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:50:49 2022

@author: ichida

1)コーパスにあるデータを用いて、sentencepieceによるトークナイザーmodelを作成し、分かち書き、頻出順上位を抽出
"""

from sentencepiece import SentencePieceTrainer
import sentencepiece as spm

drive_dir="D:/BERT/Edogawa_BERT/keikakiroku/"

# # vocab_sizeはエラーが出ない程度に大きいほうが熟語などが登録されるのでベター
SentencePieceTrainer.Train(
    '--input='+drive_dir+'corpus0122/corpus_cleaned.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'edogawa_sp --character_coverage=0.9995 --vocab_size=20000 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
    # '--input='+drive_dir+'corpus/corpus.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=1201 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
)




from transformers import AlbertTokenizer,BertJapaneseTokenizer

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去される。 
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'edogawa_sp.model', keep_accents=True)
text = "児童相談所に本児は保護されたが、名前を言えない"
print(tokenizer.tokenize(text))

corpus = open(drive_dir+"corpus0122/corpus_cleaned.txt", "r",encoding="utf-8_sig")
c = corpus.read()

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

        

"""
２）既存の語彙ファイルvocab.txtに含まれない単語数をカウント
"""

CC_list = vocab_file

txt_file = open("D:/Edogawa/model_save/protection/token/vocab.txt", "r",encoding="utf-8_sig")
txt_A = txt_file.read()

A_list = txt_A.split("\n")

C_set = set(CC_list).difference(A_list)
#Type change, Set into list
C_list = list(C_set)




"""
3)adding words to the tokenizer
"""


#語彙の追加
from transformers import AlbertTokenizer,BertJapaneseTokenizer

jtk_a = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token/")

jp_text = "内夫に叩かれた児は児童相談所に一時保護されたが、名前を言えない"

print(jtk_a.tokenize(jp_text))
print("vocab size", len(jtk_a))

jtk_a.add_tokens(C_list)
# jtk_a.add_tokens(["児童相談所"])
# jtk_a.add_tokens(["本児","児相","内夫"])


print(jtk_a.tokenize(jp_text))
print("vocab size", len(jtk_a))


jtk_a.save_pretrained('D:/Edogawa/model_save/protection/token_edo_0330/')






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
