   # -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:59:28 2021

@author: ichida


Tohokuは、Mecabにより。
Transformers==4.16.2 @Deep2　
実行するまえにGPUを空にするために、spyderを一度再起動する。

３月１５日、３月21日に追加学習成功（語彙は東北大のまま） 
江戸川区用に語彙を追加したモデルでは３月22日に追加学習成功

比較のためにepoch20で実施した、
D:\BERT\Edogawa_BERT\tohoku&edogawa03220　 
D:\Edogawa\model_save\protection\token_edo_0319 "vocab_size": 33583　
と学習後に比較する。
"""


# from sentencepiece import SentencePieceTrainer
# import sentencepiece as spm
import apex
from apex import amp,optimizers
import warnings
warnings.simplefilter('ignore')


drive_dir='D:/BERT/Edogawa_BERT/edo1.5m/'
corpus_dir='D:/BERT/Edogawa_BERT/keikakiroku/corpus0319/corpus_with_blankline2.txt' 

from transformers import AutoTokenizer, AutoModelForMaskedLM, AlbertTokenizer,BertJapaneseTokenizer
# tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token")
tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token_edo_0408/")
# tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

# model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
# リスタートの場合もここにモデルパスを渡す
model = AutoModelForMaskedLM.from_pretrained("D:\BERT\Edogawa_BERT\edo1.5m/")

# model = AutoModelForMaskedLM.from_pretrained("D:/BERT/Edogawa_BERT/tohoku&edogawa0327/checkpoint-83575 loss 0.974/")



text = "内夫に叩かれた本児は児童相談所に一時保護されたが、名前を言うことができない。"
print(tokenizer.tokenize(text))
                         
from transformers import BertConfig
from transformers import BertForMaskedLM,BertModel

# 2022 0306 使わない(基本的に読み込んだconfigを変更しない)
# config = BertConfig(vocab_size=34000, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12, max_position_embeddings=520)
# config = BertConfig(vocab_size=32000, num_hidden_layers=12, intermediate_size=3072,num_attention_heads=12, max_position_embeddings=512)
# config = BertConfig(vocab_size=len(tokenizer), num_hidden_layers=12, intermediate_size=768, num_attention_heads=12, max_position_embeddings=514,type_vocab_size=1)


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
# torch.cuda.is_available()
print("torch.cuda.is_available=",torch.cuda.is_available())

#cuda change from 11.5 to 11.3
#!nvidia-smi from 496.13 to 465.89

# modeldir = "D:/Edogawa/Edogawa_BERT/tohoku+edotoken/model0406/checkpoint-32225 0.824"

# model = BertModel.from_pretrained(modeldir)
# model = BertForMaskedLM.from_pretrained(modeldir)
# model = BertForMaskedLM(config)


##0321
model.resize_token_embeddings(len(tokenizer))
# The new vector is added at the end of the embedding matrix

# model.bert.embeddings.word_embeddings.weight[-1, :]
# # Randomly generated matrix

# model.bert.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size])
##0321


from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=corpus_dir,
     # block_size=16, 
      block_size=128, 
)

#評価用データ追加 
dataset_eva = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=drive_dir + 'eval_data/eval_parerent_train.csv',
     block_size=256, # tokenizerのmax_length
)
# 以上

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
    output_dir= drive_dir,
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=64,
    # per_device_train_batch_size=16,
    save_steps=25,
    save_total_limit=500,
    prediction_loss_only=False,
    fp16=True,
    # additional
    # resume_from_checkpoint=modeldir,#うまく反映されない、予測力がゼロになる。
    disable_tqdm=False,
    # dataloader_num_workers = 8 ,
    logging_dir= drive_dir + 'logs/',
    logging_steps=25,
    # eval_steps=100,
    # evaluation_strategy='steps',#評価用に必要  'steps'
    # save_strategy='steps',#評価用に必要
    # load_best_model_at_end=True #最後に最良のモデルを保存
    # learning_rate = 1e-7,
    # adam_beta2 = 0.8,
    # warmup_steps = 2000
)

    # warmup_steps=500,                # 学習率スケジューラのウォームアップステップ数
    # weight_decay=0.01,               # 重み減衰の強さ    
    # evaluation_strategy="epoch"
    #training_loss減らなくなる
    # dataloader_num_workers

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # eval_dataset=dataset_eva,#評価用に必要
    # eval_dataset= drive_dir + 'corpus/corpus.csv'
)

# trainer.train(drive_dir + 'EBP_BERT/checkpoint-2270000')
# trainer.train(modeldir)


trainer.train() 
trainer.save_model(drive_dir)
# tokenizer.save_pretrained('D:/Edogawa/Edogawa_BERT/tohoku+edotoken/tokenizer/')

import locale
locale.setlocale(locale.LC_CTYPE, "Japanese_Japan.932")

import datetime
dt_now = datetime.datetime.now()
dt_now_st =dt_now.strftime('%Y年%m月%d日%H-%M-%S')
# dt_now_st =str(dt_now.month)+str(dt_now.day)+str(dt_now.hour)+str(dt_now.minute)

import os
new_dir_path = 'D:/BERT/Edogawa_BERT/edo1.5m_backup/'
# os.mkdir(new_dir_path+dt_now_st)

import shutil
shutil.copytree(drive_dir,new_dir_path+dt_now_st)