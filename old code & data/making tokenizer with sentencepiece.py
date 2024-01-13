# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:56:01 2022

@author: ichida
"""

from sentencepiece import SentencePieceTrainer
import sentencepiece as spm
drive_dir='C:/Users/owner/Desktop/BERT/keikakiroku/'



# vocab_sizeはエラーが出ない程度に大きいほうが熟語などが登録されるのでベター
SentencePieceTrainer.Train(
    # '--input=corpus_dir, --train_extremely_large_corpus --model_prefix='+drive_dir+'edogawa_sentencepiece --character_coverage=0.9995 --vocab_size=40000 --pad_id=3 --add_dummy_prefix=False'
    '--input='+drive_dir+'corpus_wiki+edogawa.txt, --train_extremely_large_corpus=true --model_prefix='+drive_dir+'edogawa_sentencepiece --character_coverage=0.9995 --vocab_size=31000 --pad_id=3 --add_dummy_prefix=False --max_sentence_length=256'
    # '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'edogawa_sentencepiece --character_coverage=0.9995 --vocab_size=406 --pad_id=3 --add_dummy_prefix=False --max_sentence_length'
    # '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'edogawa_sentencepiece --character_coverage=0.9995 --vocab_size=6 --pad_id=3 --model_type=word --user_defined_symbols=<SEP>,<CLS> --add_dummy_prefix=False'
    # '--input='+drive_dir+'corpus/corpus.txt, --model_prefix='+drive_dir+'edogawa_sentencepiece --vocab_size=24 --model_type=word --user_defined_symbols="_--user_defined_symbols=<SEP>--user_defined_symbols=<SEP>"'
)


from transformers import AlbertTokenizer

# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'edogawa_sentencepiece.model', keep_accents=True)
text = "児童相談所への通告により、本児は一時保護となった"
print(tokenizer.tokenize(text))

sp = spm.SentencePieceProcessor()
sp.Load(drive_dir+'edogawa_sentencepiece.model')
print(sp.EncodeAsPieces(text))