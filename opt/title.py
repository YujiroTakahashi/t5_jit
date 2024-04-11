#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import pprint
import torch
from transformers import T5Tokenizer

pp = pprint.PrettyPrinter(indent=4, compact=True, width=120)
model_path = "/opt/model/t5-qiita-title-generation"

tokenizer = T5Tokenizer.from_pretrained(model_path, is_fast=True)
model = torch.jit.load("traced_jit.pt", map_location="cpu")

# ノーマライズされたテキスト
text = """
そんな折、縁あってインターネットサービスを提供している会社の方との接点を持つ機会があり、インターネットの可能性に魅了されました。時間も場所も距離もコストも取っ払い、人と人、人と作品、人と企業、人とデータ、人とモノ、企業と企業、企業とデータetc...をあらゆる点をつなげることができれば、指数関数的に"誰か"の笑顔を生み出し続けることができるではないか、と。
そんな思いからインターネットの世界に身を移し、現在は「EDITECH（EDIT+TECHNOLOGY）」を掲げるコンテンツソリューションプロバイダとして、50万人を超えるライターの方々と累計4,000社を超えるお取引先様に恵まれるまでに成長することができました。
「誰か」のために動いていたら、CROCOとして対峙する人数は社会人になりたての頃に比べ数万倍になり、営業からスタートしたキャリアは代表取締役社長という立場になっていました。
"""

batch = tokenizer([text], padding="longest", return_tensors="pt")

model.eval()

with torch.no_grad():
    outputs = model.generate(batch["input_ids"])
    text = tokenizer.decode(outputs[0], skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False) 
    print(text)

