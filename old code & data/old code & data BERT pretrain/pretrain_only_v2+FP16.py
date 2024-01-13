# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:59:28 2021

@author: ichid

transformers                       4.15.0

"""
from sentencepiece import SentencePieceTrainer

# drive_dir='//EBP-NAS01/edogawa/BERT/'
drive_dir='C:/wikipedia_corpus/'


SentencePieceTrainer.Train(
    '--input='+drive_dir+'corpus0124.txt, --model_prefix='+drive_dir+'sentencepiece --character_coverage=0.9995 --vocab_size=300 --pad_id=3 --add_dummy_prefix=False'
)

# SentencePieceTrainer.Train(
#     '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'sentencepiece --character_coverage=0.9995 --vocab_size=32000 --pad_id=3 --add_dummy_prefix=False'
# )

from transformers import AlbertTokenizer

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'sentencepiece.model', keep_accents=True)
text = "吾輩は猫である。名前はまだ無い。"
print(tokenizer.tokenize(text))

# Adding [CLS] to the vocabulary
# Adding [SEP] to the vocabulary
# Adding [MASK] to the vocabulary
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# ['吾輩は猫である', '。', '名前は', 'まだ', '無い', '。']


from apex import amp, optimizers
# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()


from transformers import BertConfig
from transformers import BertForMaskedLM

config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
model = BertForMaskedLM(config)



from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=drive_dir + 'corpus0124.txt',
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
    output_dir= drive_dir + 'BERTmodel/',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=100,
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

# trainer.train(resume_from_checkpoint=drive_dir + 'BERTmodel/')


trainer.save_model(drive_dir + 'BERTmodel/')


