# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:59:28 2021

@author: ichid
"""


from sentencepiece import SentencePieceTrainer
import sentencepiece as spm
drive_dir='C:/Users/owner/Desktop/BERT/'
# corpus_dir=drive_dir+'corpus/corpus.txt'

#vocab_size in BERT must be larger than that of SentencePieceTrainer


# vocab_sizeはエラーが出ない程度に大きいほうが熟語などが登録されるのでベター
SentencePieceTrainer.Train(
    # '--input=corpus_dir, --train_extremely_large_corpus --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=40000 --pad_id=3 --add_dummy_prefix=False'
    '--input='+drive_dir+'corpus/kaigyozumi/AA.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=10000 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
    # '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=406 --pad_id=3 --add_dummy_prefix=False --max_sentence_length'
    # '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=6 --pad_id=3 --model_type=word --user_defined_symbols=<SEP>,<CLS> --add_dummy_prefix=False'
    # '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --vocab_size=24 --model_type=word --user_defined_symbols="_--user_defined_symbols=<SEP>--user_defined_symbols=<SEP>"'
)


from transformers import AlbertTokenizer

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'souseki_sentencepiece.model', keep_accents=True)
text = "アメリカに私が行ったが名前はまだ無い"
print(tokenizer.tokenize(text))

sp = spm.SentencePieceProcessor()
sp.Load(drive_dir+'souseki_sentencepiece.model')
print(sp.EncodeAsPieces(text))


# tokenizer.decode(encoded_input["input_ids"])
# "[CLS] Hello, I'm a single sentence! [SEP]"


# Adding [CLS] to the vocabulary
# Adding [SEP] to the vocabulary
# Adding [MASK] to the vocabulary
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# ['吾輩は猫である', '。', '名前は', 'まだ', '無い', '。']




corpus_dir=drive_dir+'corpus/kaigyozumi/corpus.txt' #0101 0639 START
                         
from transformers import BertConfig
from transformers import BertForMaskedLM

#BERT vocab_size
# config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
# # adjusing to Kyoto
# config = BertConfig(vocab_size=32006, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)
# adjusing to Tokyo
# config = BertConfig(vocab_size=30000, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)
# 2021 12 31final
config = BertConfig(vocab_size=32000, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
# torch.cuda.is_available()
print("torch.cuda.is_available=",torch.cuda.is_available())

#cuda change from 11.5 to 11.3
#!nvidia-smi from 496.13 to 465.89


model = BertForMaskedLM(config)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=corpus_dir,
     block_size=256, 
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

training_args = TrainingArguments(
    output_dir= drive_dir + 'SousekiBERT/',
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=32,
    # per_device_train_batch_size=16,
    save_steps=5000,
    save_total_limit=3,
    prediction_loss_only=True,
    # additional
    disable_tqdm=False,
    logging_dir= drive_dir + 'logs/',
    logging_steps=1
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

# trainer.train()             
trainer.train(drive_dir + 'SousekiBERT/')
trainer.save_model(drive_dir + 'SousekiBERT/')



# checkpoint='C:/Users/owner/Desktop/BERT/SousekiBERT/'
# trainer.train(checkpoint)





# from transformers import AlbertTokenizer
# from transformers import pipeline
# from transformers import BertForMaskedLM
# # from sudachitra import BertSudachipyTokenizer
# # from sudachipy import tokenizer

# drive_dir='C:/Users/ichid/Documents/GitHub/BERT/'

# tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'souseki_sentencepiece.model', keep_accents=True)
# model = BertForMaskedLM.from_pretrained(drive_dir + 'SousekiBERT')

# # tokenizer = BertSudachipyTokenizer.from_pretrained('sudachitra-bert-base-japanese-sudachi')
# # model = BertForMaskedLM.from_pretrained(drive_dir + 'SousekiBERT')


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
