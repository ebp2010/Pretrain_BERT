# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:46:31 2022

@author: owner2

20250221 
conda pretrain@Deep2
"""

"""
穴埋め問題
"""

import torch

from transformers import BertJapaneseTokenizer
# from transformers import pipeline
from transformers import BertForMaskedLM
# from sudachitra import BertSudachipyTokenizer
# from sudachipy import tokenizer

# tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token")
# model = BertForMaskedLM.from_pretrained("D:/Edogawa/model_save/protection/model/")

from transformers import AutoTokenizer, AutoModelForMaskedLM

tohoku ="D:/BERT/agri_cost_adding_tokens/tohoku_model"

tokenizer = AutoTokenizer.from_pretrained(tohoku)

model = AutoModelForMaskedLM.from_pretrained(tohoku)

# from transformers import pipeline


def fill_mask(text, model, tokenizer, top_k=5):
    """
    text: 入力テキスト。マスク部分には tokenizer.mask_token を含める
    model: AutoModelForMaskedLM などの事前学習済みマスクド言語モデル
    tokenizer: 対応するトークナイザー
    top_k: 上位何件の候補を返すか
    """
    # 入力テキストをトークン化し、テンソルに変換
    inputs = tokenizer(text, return_tensors="pt")
    
    # マスクトークンの位置を取得（ここでは1つのマスクしかない前提）
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if mask_token_index[0].numel() == 0:
        raise ValueError("入力文にマスクトークンが見つかりません")
    # 複数あった場合は最初のものを使用（必要に応じて拡張可能）
    mask_index = mask_token_index[1][0]
    
    # モデル推論（評価モードにしておく）
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # shape: [batch_size, sequence_length, vocab_size]
    
    # マスクトークンの位置におけるロジットを取り出し、上位top_kの候補を取得
    mask_logits = logits[0, mask_index, :]
    topk = torch.topk(mask_logits, top_k)
    top_tokens = topk.indices.tolist()
    top_scores = topk.values.tolist()
    
    results = []
    for token_id, score in zip(top_tokens, top_scores):
        token_str = tokenizer.decode([token_id]).strip()
        # マスク部分を候補トークンで置換した文章を作成
        output_text = text.replace(tokenizer.mask_token, token_str)
        results.append({
            "score": score,
            "token_str": token_str,
            "sequence": output_text
        })
    return results


anaume_bun = "コンバインは{}に使う機械です。"

sample_text = anaume_bun.format(tokenizer.mask_token)
predictions = fill_mask(sample_text, model, tokenizer, top_k=5)

for pred in predictions:
    print(f"候補: {pred['token_str']}, スコア: {pred['score']}")
    print(f"生成文: {pred['sequence']}\n")

# fill_mask = pipeline(
#     "fill-mask",
#     model=model,
#     tokenizer=tokenizer
# )

# MASK_TOKEN = tokenizer.mask_token

# text = '''
# 草刈り機で{}をした。
# '''.format(MASK_TOKEN)
# fill_mask(text)




# from transformers import pipeline
tokenizer = BertJapaneseTokenizer.from_pretrained("D:/BERT/agri_cost_adding_tokens/agri_costBERT0221tokenizer/")
# tokenizer = BertJapaneseTokenizer.from_pretrained("D:/Edogawa/model_save/protection/token")
# tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model = BertForMaskedLM.from_pretrained("D:/BERT/agri_cost_adding_tokens/agri_costBERT0221/")


sample_text = anaume_bun.format(tokenizer.mask_token)
predictions = fill_mask(sample_text, model, tokenizer, top_k=5)

for pred in predictions:
    print(f"候補: {pred['token_str']}, スコア: {pred['score']}")
    print(f"生成文: {pred['sequence']}\n")

# fill_mask = pipeline(
#     "fill-mask",
#     model=model,
#     tokenizer=tokenizer
# )

# MASK_TOKEN = tokenizer.mask_token

# text = '''
# 草刈り機で{}をした。
# '''.format(MASK_TOKEN)
# fill_mask(text)
