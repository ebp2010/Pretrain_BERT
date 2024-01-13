# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:59:28 2021

@author: ichida


Tohokuは、Mecabにより。
Transformers==4.16.2 @Deep2　で３月１５日、３月22日に追加学習成功（語彙は東北大のまま） 

＊＊＊adamの設定について＊＊＊
3月２２日までに追加学習によりlossは1.2程度まで減少。
3月２６日同じ設定で開始したが、lossは上昇。
3月２６日に learning_rate = 5e-06に小さくして再開したところ(デフォルト5e-05）、
lossが約1.0まで減少したが、途中から上昇。

beta1は0.9デフォルトで、過去の情報の反映割合を示す。
beta2は0.999デフォルトで、振動が大きくなった時に学習率を小さくする程度を示す。0.7へ低下なら学習率も結局低下

＊＊＊注意点＊＊＊
①実行するまえにGPUを空にするために、spyderを一度再起動する。
②以下にモデルのパスを入れたほうが、checkpointであっても追加学習しやすい。lossに連続性あるか確認。
 model = AutoModelForMaskedLM.from_pretrained(***)




"""
import warnings
warnings.simplefilter('ignore')

# from sentencepiece import SentencePieceTrainer
# import sentencepiece as spm
import apex
drive_dir='D:/Edogawa/Edogawa_BERT/tohoku+edotoken/'
# corpus_dir=drive_dir+'corpus/corpus.txt'

#vocab_size in BERT must be larger than that of SentencePieceTrainer


# # vocab_sizeはエラーが出ない程度に大きいほうが熟語などが登録されるのでベター
# SentencePieceTrainer.Train(
#     '--input='+drive_dir+'corpus/kaigyozumi/corpus.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=10000 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
#     # '--input='+drive_dir+'corpus/corpus.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=1201 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
# )


from transformers import AlbertTokenizer,BertJapaneseTokenizer

# model_name_or_path = "cl-tohoku/bert-base-japanese-v2"
# tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token")
tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token_edo_0319")
# tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")


from transformers import AutoTokenizer, AutoModelForMaskedLM

#途中からでもこの形で開始する。training_argsに書き込まない。
# modeldir = "D:/Edogawa/Edogawa_BERT/tohoku+edotoken2/"
# model = AutoModelForMaskedLM.from_pretrained(modeldir)


model = AutoModelForMaskedLM.from_pretrained("D:/Edogawa/Edogawa_BERT/tohoku&edogawa0327/checkpoint-166500")



# tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'tokenizer/souseki_sentencepiece.model', keep_accents=True, do_lower_case=False)
# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。 
# tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'tokenizer/souseki_sentencepiece.model', keep_accents=True)
text = "内夫に叩かれた本児は児童相談所に一時保護されたが、名前を言えない"
print(tokenizer.tokenize(text))

# sp = spm.SentencePieceProcessor()
# sp.Load(drive_dir+'tokenizer/souseki_sentencepiece.model')
# print(sp.EncodeAsPieces(text))



corpus_dir='D:/Edogawa/Edogawa_BERT/keikakiroku/corpus0319/corpus_with_blankline.txt' 
                         
from transformers import BertConfig
from transformers import BertForMaskedLM,BertModel

#BERT vocab_size
# config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
# # adjusing to Kyoto
# config = BertConfig(vocab_size=32006, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)
# adjusing to Tokyo
# config = BertConfig(vocab_size=30000, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)

# 2022 0306

# config = BertConfig(vocab_size=34000, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12, max_position_embeddings=520)
config = BertConfig(vocab_size=32000, num_hidden_layers=12, intermediate_size=3072,num_attention_heads=12, max_position_embeddings=512)
# config = BertConfig(vocab_size=len(tokenizer), num_hidden_layers=12, intermediate_size=768, num_attention_heads=12, max_position_embeddings=514,type_vocab_size=1)



import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
# torch.cuda.is_available()
print("torch.cuda.is_available=",torch.cuda.is_available())

#cuda change from 11.5 to 11.3
#!nvidia-smi from 496.13 to 465.89


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
    num_train_epochs=100,
    # learning_rate = 5e-06, #loss=1.2から再開の設定　5e-05より小さく 
    learning_rate = 9e-06, #loss=1.0から再開の設定　5e-05より小さく
    adam_beta1 = 0.8,#loss=1.0から再開の設定　0.9より大きくした（現在志向）
    adam_beta2 = 0.99,#loss=1.0から再開の設定　0.999より大きくした（振動したら学習率小さく）
    # per_device_train_batch_size=256,
    per_device_train_batch_size=64,
    # per_device_train_batch_size=16,
    save_steps=25,
    save_total_limit=40,
    # load_best_model_at_end=True,
    # evaluation_strategy = "steps",
    # save_strategy = "steps",    
    prediction_loss_only=True,
    # fp16=True,
    # additional
    # resume_from_checkpoint="D:/Edogawa/Edogawa_BERT/tohoku+edotoken/checkpoint-249000",
    disable_tqdm=False,
    # dataloader_num_workers = 8 ,
    logging_dir= drive_dir + 'logs/',
    logging_steps=25
)

    # warmup_steps=500,                # 学習率スケジューラのウォームアップステップ数
    # weight_decay=0.01,               # 重み減衰の強さ    
    # evaluation_strategy="epoch"
    #training_loss減らなくなる
    # dataloader_num_workers
    # load_best_model_at_end
    


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # eval_dataset= drive_dir + 'corpus/corpus.csv'
)


   
# trainer.train(drive_dir + 'EBP_BERT/checkpoint-2270000')
# trainer.train(modeldir)

trainer.train() 

trainer.save_model('D:/Edogawa/Edogawa_BERT/tohoku+edotoken/')

# tokenizer.save_pretrained('D:/Edogawa/Edogawa_BERT/tohoku+edotoken/tokenizer/')









