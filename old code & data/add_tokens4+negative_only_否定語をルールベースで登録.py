
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:50:49 2022

@author: ichida

1)コーパスを読み込み、sentencepieceによるトークナイザーmodelを作成し、分かち書き、頻出順上位を抽出

py37TF4162@deep2
"""


# 以下はすでに実施しファイル作成済み

# corpus = open(drive_dir+"corpus0122/corpus_cleaned.txt", "r",encoding="utf-8_sig")
# ccc = corpus.read()

# import neologdn
# cc = neologdn.normalize(ccc)

# import unicodedata
# c = unicodedata.normalize("NFKC", cc)

# with open(drive_dir+"corpus0122/corpus_cleaned_std.txt", mode='w',encoding="utf-8_sig") as f:
#     f.write(c)



# from sentencepiece import SentencePieceTrainer
# # import sentencepiece as spm



# # # vocab_sizeはエラーが出ない程度に大きいほうが熟語などが登録されるのでベター
# SentencePieceTrainer.Train(
#     '--input='+drive_dir+'corpus0122/corpus_cleaned_std.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'edogawa_sp --character_coverage=0.9995 --vocab_size=25000 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
#     # '--input='+drive_dir+'corpus/corpus.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=1201 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
# )




# from transformers import AlbertTokenizer,BertJapaneseTokenizer

# # AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去される。 
# tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'edogawa_sp.model', keep_accents=True)
# text = "児童相談所に本児は保護されたが、名前を言えない"
# print(tokenizer.tokenize(text))

# c_t = tokenizer.tokenize(c)


# words = c_t

# import collections
# import regex
# import re

# c_c = collections.Counter(words)
# c_best = c_c.most_common(80000)

# vocab_file = []
# for i in c_best:
#     ii=i[0]
#     ii = re.sub(r"\d","",ii)
#     # if len(ii)>1:
#     rii = regex.findall("\p{Han}+", ii)
#     if rii!=[] and len(ii)==len(rii[0]) and len(ii)>1:
#         print(ii," ",rii)
#         vocab_file.append(ii)
        
# for i in c_best:
#     ii=i[0]
#     ii = re.sub(r"\d","",ii)
#     # if len(ii)>1:
#     rii = regex.findall("\p{Katakana}+", ii)
#     if rii!=[] and len(ii)==len(rii[0]) and len(ii)>1 and ii!="":
#         print(ii," ",rii)
#         vocab_file.append(ii)    


# # for i in c_best:
# #     ii=i[0]
# #     ii = re.sub(r"\d","",ii)
# #     # if len(ii)>1:
# #     rii = regex.findall('[a-zA-Zａ-ｚＡ-Ｚ]+', ii)
# #     # rii = regex.findall('[Ａ-Ｚ]+', ii)
# #     if rii!=[] and len(ii)==len(rii[0]) and len(ii)>1:
# #         print(ii," ",rii)
# #         vocab_file.append(ii)   

# # cp = c.split()

# #### 匿名化の結果、名前がAAなどのアルファベットになっているため、利用困難
# # eng = []
# # for word in c.split():
# #     word = re.findall('[A-Za-z]+', word)
# #     if word and len(word[0])>2 and "XX" not in word[0]and len(word[0])<10:
# #         print(word[0])
# #         eng.append(word[0])
    
# # S = sorted(list(set(eng)))
# # print(S)
# # vocab_file = vocab_file + S
# # print(vocab_file)

vocab_dir="D:/Edogawa/model_save/protection/token/"

"""
２）既存の語彙ファイルvocab.txtに含まれない単語数をカウント
"""
CC_list = ['かもしれない','わからない','分からない','いけない','ことはない','ならない','ざるを得ない','ざるをえない','いられない','かまわない','構わない','たまらない','てはいけない','いられない','ではないか','ならない','もらえない','ったらない','どころではない','過ぎない','足りない','違いない','ちがいない','とどまらない','及ばない','およばない','ほかならない','べくもない','こともない','までもない','ほかない','できない','出来ない','仕方ない','ない','ぬ','ん','非','不','無','未','反','異','無い','無し','物足りない','仕方がない','思わず','なければならない','ないといけない','かもしれません','にもかかわらず','だけでなく']

#BERTに含まれるvocab.txtを読み込み
txt_file = open(vocab_dir+"vocab.txt", "r",encoding="utf-8_sig")
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
jtk_a = BertJapaneseTokenizer.from_pretrained(vocab_dir)

jp_text = "児が、内夫に叩かれていないかはわからないが、児童相談所は児を一時保護せざるをえない。"
# jp_text = "内夫に叩かれた児は児童相談所に一時保護されたが、名前を言えない。児はベンチに座っている。"

print(jtk_a.tokenize(jp_text))
print("vocab size", len(jtk_a))

jtk_a.add_tokens(C_list)
# jtk_a.add_tokens(["児童相談所"])
# jtk_a.add_tokens(["本児","児相","内夫"])

print(jtk_a.tokenize(jp_text))
print("vocab size", len(jtk_a))

jtk_a.save_pretrained('D:/Edogawa/model_save/protection/')






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
