# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:59:28 2021

@author: ichid
"""
from sentencepiece import SentencePieceTrainer

drive_dir='C:/Users/ichid/Documents/GitHub/BERT/'

SentencePieceTrainer.Train(
    '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=829 --pad_id=3 --add_dummy_prefix=False'
)

# SentencePieceTrainer.Train(
#     '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'souseki_sentencepiece --character_coverage=0.9995 --vocab_size=32000 --pad_id=3 --add_dummy_prefix=False'
# )

from transformers import AlbertTokenizer

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'souseki_sentencepiece.model', keep_accents=True)
text = "吾輩は猫である。名前はまだ無い。"
print(tokenizer.tokenize(text))

# Adding [CLS] to the vocabulary
# Adding [SEP] to the vocabulary
# Adding [MASK] to the vocabulary
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# ['吾輩は猫である', '。', '名前は', 'まだ', '無い', '。']

from transformers import BertConfig
from transformers import BertForMaskedLM

config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
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
    mlm_probability= 0.15
)




from transformers import TrainingArguments
from transformers import Trainer

training_args = TrainingArguments(
    output_dir= drive_dir + 'SousekiBERT/',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)


trainer.train()

trainer.save_model(drive_dir + 'SousekiBERT/')



from transformers import AlbertTokenizer
from transformers import pipeline
from transformers import BertForMaskedLM

drive_dir='C:/Users/ichid/Documents/GitHub/BERT/'

tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'souseki_sentencepiece.model', keep_accents=True)
model = BertForMaskedLM.from_pretrained(drive_dir + 'SousekiBERT')

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
自分の{}の許す限りは。
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
