# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:59:28 2021

@author: ichid
"""


from sentencepiece import SentencePieceTrainer

import sentencepiece as spm

drive_dir='C:/Users/ichid/Documents/GitHub/BERT/'

SentencePieceTrainer.Train(
    # '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=555 --pad_id=3 --add_dummy_prefix=False'
    '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=7 --pad_id=3 --model_type=word --add_dummy_prefix=False'
)

# --model_type

# SentencePieceTrainer.Train(
#     '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=32000 --pad_id=3 --add_dummy_prefix=False'
# )

from transformers import AlbertTokenizer

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'souseki_sentencepiece.model', keep_accents=True)
text = "猫である名前はまだ'無い"
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



drive_dir='C:/Users/ichid/Documents/GitHub/BERT/'

from transformers import BertConfig
from transformers import BertForMaskedLM

# config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)

# # adjusing to Kyoto
# config = BertConfig(vocab_size=32006, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)

# adjusing to Tokyo
config = BertConfig(vocab_size=25000, num_hidden_layers=12, intermediate_size=3072, num_attention_heads=12)
model = BertForMaskedLM(config)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=drive_dir + 'corpus/corpus.txt',
     block_size=256, # tokenizerのmax_length
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True,
    # mlm_probability= 0.15
    mlm_probability= 0.05
)

from transformers import TrainingArguments
from transformers import Trainer

training_args = TrainingArguments(
    output_dir= drive_dir + 'SousekiBERT/',
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=1,
    prediction_loss_only=False,
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
    
    # additional
    # eval_dataset= drive_dir + 'corpus/corpus.csv'
)


trainer.train()

# trainer.evaluate(eval_dataset= drive_dir + 'corpus/corpus.csv')

trainer.train(resume_from_checkpoint=drive_dir + 'SousekiBERT/')


trainer.save_model(drive_dir + 'SousekiBERT/')



from transformers import AlbertTokenizer
from transformers import pipeline
from transformers import BertForMaskedLM
# from sudachitra import BertSudachipyTokenizer
# from sudachipy import tokenizer

drive_dir='C:/Users/ichid/Documents/GitHub/BERT/'

tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'souseki_sentencepiece.model', keep_accents=True)
model = BertForMaskedLM.from_pretrained(drive_dir + 'SousekiBERT')

# tokenizer = BertSudachipyTokenizer.from_pretrained('sudachitra-bert-base-japanese-sudachi')
# model = BertForMaskedLM.from_pretrained(drive_dir + 'SousekiBERT')


fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

MASK_TOKEN = tokenizer.mask_token
# text = '''
# 吾輩は{}である。名前はまだ無い。
# '''.format(MASK_TOKEN)

text = '''
保護{}の話を聞く
'''.format(MASK_TOKEN)
fill_mask(text)






# [{'score': 0.002911926247179508,
#  'sequence': '吾輩は自分である。名前はまだ無い。',
#  'token': 164,
#  'token_str': '自分'},
# {'score': 0.0022156336344778538,
#  'sequence': '吾輩はそれである。名前はまだ無い。',
#  'token': 193,
#  'token_str': 'それ'},
