# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:59:28 2021

@author: ichida

Transformers==4.8.0で動いた（souseki_sentencepiece.model） 
エラーが出た場合はspyderを再起動する必要あり

"""

from sentencepiece import SentencePieceTrainer
import sentencepiece as spm
import apex
drive_dir='D:/BERT/'
# corpus_dir=drive_dir+'corpus/corpus.txt'
#vocab_size in BERT must be larger than that of SentencePieceTrainer


# # vocab_sizeはエラーが出ない程度に大きいほうが熟語などが登録されるのでベター
# SentencePieceTrainer.Train(
#     '--input='+drive_dir+'corpus/kaigyozumi/corpus.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=973 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
#     # '--input='+drive_dir+'corpus/corpus.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=1201 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
# )

from transformers import AlbertTokenizer

# tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'converting_from_Tensorflow/tokenizer/wiki-ja.model', keep_accents=True, do_lower_case=False)
# tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'converting_from_Tensorflow/tokenizer/wiki-ja.model', keep_accents=True)

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。 
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'tokenizer/souseki_sentencepiece.model', keep_accents=True)
text = "アメリカに私が行ったが名前はまだ無い"
print(tokenizer.tokenize(text))

# sp = spm.SentencePieceProcessor()
# sp.Load(drive_dir+'converting_from_Tensorflow/kikuta/tokenizer/wiki-ja.model')
# print(sp.EncodeAsPieces(text))

corpus_dir=drive_dir+'corpus/kaigyozumi/corpus.txt' 
                         
from transformers import BertConfig
from transformers import BertForMaskedLM

#BERT vocab_size
# config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
# # adjusing to Kyoto
# config = BertConfig(vocab_size=32006, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)
# adjusing to Tokyo
# config = BertConfig(vocab_size=30000, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)
# 2021 12 31final
config = BertConfig(vocab_size=32000, num_hidden_layers=12, intermediate_size=3072, hidden_size=768,num_attention_heads=12,max_position_embeddings=512)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
# torch.cuda.is_available()
print("\ntorch.cuda.is_available=",torch.cuda.is_available())

#cuda change from 11.5 to 11.3
#!nvidia-smi from 496.13 to 465.89


model = BertForMaskedLM(config)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=corpus_dir,
     block_size=32, 
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True,
    mlm_probability= 0.15
    # mlm_probability= 0.05
)


from transformers import TrainingArguments
from transformers import Trainer

import torch
torch.backends.cuda.matmul.allow_tf32 = True

training_args = TrainingArguments(
    output_dir= drive_dir + 'EBP_BERT/',
    overwrite_output_dir=True,
    num_train_epochs=3,
    # per_device_train_batch_size=32,
    # per_device_train_batch_size=64,
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=3,
    prediction_loss_only=True,
    fp16=True,
    # additional
    disable_tqdm=False,
    logging_dir= drive_dir + 'logs/',
    logging_steps=10
)
    # per_device_eval_batch_size=64,   # 評価のバッチサイズ
    # warmup_steps=500,                # 学習率スケジューラのウォームアップステップ数
    # weight_decay=0.01,               # 重み減衰の強さ    
    # evaluation_strategy="epoch"
    #training_loss減らなくなる
    # logging_steps=5
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # eval_dataset= drive_dir + 'corpus/corpus.csv'
)


         
trainer.train(drive_dir + 'converting_from_Tensorflow/kikuta')

# trainer.train()    
trainer.save_model(drive_dir + 'converting_from_Tensorflow')


# checkpoint='C:/Users/owner/Desktop/BERT/EBP_BERT/'
# trainer.train(checkpoint)


jtk_a = tokenizer
jp_text = "航空宇宙産業 (Aerospace Industry) とは、航空機や航空機の部品、ミサイル、ロケット、宇宙船を製造する産業である。この産業には、設計、製造、テスト、販売、整備などの工程がある。その規模が大きければ部分的に関わる企業、組織が存在する。本項では、エアロスペース・マニュファクチャー（英語: Aerospace manufacturer）についても述べる。"

print(jtk_a.tokenize(jp_text))
print("vocab size", len(jtk_a))


jtk_a.add_tokens(["児童相談所"])

print(jtk_a.tokenize(jp_text))
print("vocab size", len(jtk_a))

jtk_a.save_pretrained(drive_dir + 'converting_from_Tensorflow/tokenizer_txt/')


# from transformers import AlbertTokenizer
# from transformers import pipeline
# from transformers import BertForMaskedLM
# # from sudachitra import BertSudachipyTokenizer
# # from sudachipy import tokenizer

# drive_dir='C:/Users/ichid/Documents/GitHub/BERT/'

# tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'souseki_sentencepiece.model', keep_accents=True)
# model = BertForMaskedLM.from_pretrained(drive_dir + 'EBP_BERT')

# # tokenizer = BertSudachipyTokenizer.from_pretrained('sudachitra-bert-base-japanese-sudachi')
# # model = BertForMaskedLM.from_pretrained(drive_dir + 'EBP_BERT')


# fill_mask = pipeline(
#     "fill-mask",
#     model=model,
#     tokenizer=tokenizer
# )

# MASK_TOKEN = tokenizer.mask_token
# # text = '''
# # 吾輩は{}である。名前はまだ無い。
# # '''.format(MASK_TOKEN)

# text = '''
# 保護{}の話を聞く
# '''.format(MASK_TOKEN)
# fill_mask(text)

# # [{'score': 0.002911926247179508,
# #  'sequence': '吾輩は自分である。名前はまだ無い。',
# #  'token': 164,
# #  'token_str': '自分'},
# # {'score': 0.0022156336344778538,
# #  'sequence': '吾輩はそれである。名前はまだ無い。',
# #  'token': 193,
# #  'token_str': 'それ'},
