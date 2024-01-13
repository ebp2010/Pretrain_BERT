import pandas as pd
import jaconv
from sentencepiece import SentencePieceTrainer
import sentencepiece as spm
from transformers import AlbertTokenizer,BertJapaneseTokenizer
import collections
import regex
import re

# open the text file 
drive_dir="Su_Myat_san/20220420_Alphabet/"
text_file = open(drive_dir+"corpus.txt", "r",encoding='UTF-8')

# read the file 
c = text_file.read()

# to include full width alphabet from the text file
full_width_result = jaconv.normalize(c,'NFKC')

# to tokenize words from text file that include full width alphabet
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'edogawa_sp.model', keep_accents=True)
c_t = tokenizer.tokenize(full_width_result)

words = c_t

c_c = collections.Counter(words)
c_best = c_c.most_common(80000)

vocab_file = []
for i in c_best:
    ii=i[0]
    ii = re.sub(r"\d","",ii)
    rii = regex.findall("\p{Han}+", ii)
    if rii!=[] and len(ii)==len(rii[0]) and len(ii)>1:
        vocab_file.append(ii)
        
        
# add alphabetical words into vocab file
for j in full_width_result.split():
    jj = regex.findall('[A-Za_z]+', j)
    if jj!=[]:
        vocab_file.append(jj[0])
print(vocab_file)

# save the result as text file
with open('Su_Myat_san/20220420_Alphabet/fullwidth_alphabet.txt', 'w',encoding='UTF-8') as f:
    for item in vocab_file:
        f.write("%s\n" % item)
