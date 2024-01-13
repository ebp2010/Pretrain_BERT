# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:46:31 2022

@author: owner2
"""

"""
穴埋め問題
"""



from transformers import BertJapaneseTokenizer
from transformers import pipeline
from transformers import BertForMaskedLM
# from sudachitra import BertSudachipyTokenizer
# from sudachipy import tokenizer

# tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token")
# model = BertForMaskedLM.from_pretrained("D:/Edogawa/model_save/protection/model/")

from transformers import AutoTokenizer, AutoModelForMaskedLM


tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

MASK_TOKEN = tokenizer.mask_token

text = '''
内夫が児童を暴行していたため、江戸川区の{}が一時保護した。
'''.format(MASK_TOKEN)
fill_mask(text)


from transformers import pipeline
tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/Edogawa_BERT/tohoku+edotoken/tokenizer/")
# tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token")
# tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model = BertForMaskedLM.from_pretrained("D:/Edogawa/Edogawa_BERT/tohoku+edotoken/")

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

MASK_TOKEN = tokenizer.mask_token

text = '''
内夫が児童を暴行していたため、江戸川区の{}が一時保護した。
'''.format(MASK_TOKEN)
fill_mask(text)
