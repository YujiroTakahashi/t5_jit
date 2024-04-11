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

input_ids = torch.tensor([[    5,  5802, 15883,  3862,  5286,     3,  9650, 26726,  4824,  8015,
            86,  1475,  3862, 20281,    22,   947,  2970,     3,  7180, 10595,
          2509,   676,  2730,  6470,   611,  9762,  2577,     4, 13712,   801,
            13,     3,  3862, 20281,  9463, 24599, 26916,  8054, 26726,   351,
           240, 23058,   193,  2124,     4, 26488,    93, 14803,    19,    93,
         14803,     7,     3,  7618,  8588,   611,  9762,    47,    89,     3,
           451,  1613,    22,   451,  8697,   337,   139,     8,   306,  1869,
            58, 26916,     4,  4134,     3,  1356,  1925,   429,     7,   742,
           322, 24599,  7828,  1589,  8054,   296,   323,   606,    65,  1004,
            11,  7618,  2067,     3,   451,  1613,   105, 16328,  1038,    93,
         14803,     4,  3669,  4079,    47,  2942,     3,   327,  1914,    14,
           369,   247,   725,    16,   451,   304,    53,  4459,   801,     7,
           142,  9260,  2090,   876, 22160,     4, 26488, 20237,    19,  1030,
           875,  4194,  5395,    13,   987, 12492,    16,   403,   780,     8,
         27193,    15,  2764,  2963,    32,    16, 29904,  1038,     6,     7,
          4281,   780,    14,    22,   947,   213, 21369,     4,   699,  7175,
          3848,  4456,    13,   780, 12492,  1493,    28,   211,   165,   365,
           296,  1465,   606,    65,  1834,    11,  4902,    22, 11620,   774,
            15,   987,    14,  1732,   209,  4985, 12339,  3375,  1464,  8660,
             3,   220,  1861,    15,   987,    14,   339,    86,  3428,  4453,
            28, 13837, 10551,     4,  9817,  2942,   944,  7043, 27193,    53,
           987,    14,   339, 11689,    82,   287,  9970,  8597,    28,   253,
          1475, 26488, 11592,    19,  9866,     6,  5286,     3, 15721,  5772,
          3215,    13, 19772,     4, 11592,    15, 13976, 23836,   584,  1525,
            86,     3, 17617,  1902,   397,  9684,  3797,  4339,     4,  3876,
             3,  9363,    13,     8,  1288,    29, 13282, 11752,     8, 14355,
             3,   417,   307,   163,  9747,     6,     8, 15878,  7533,   296,
           323,   606,    65, 21254,    11,  9363,   548,    23, 11592,     7,
             3,  5037,  4450,  8685, 23836,  1259,   998,     4,  3862,  5286,
         16103, 11752,   322,     3,  1502,   313,   134,  3647,   769, 10950,
         11286, 22160,     4, 26488,   635,  5593,    19,   635,  5593,    13,
          1005,    12,    28, 19772,     6,    15,     3,  1405,  2696,    28,
         16215, 15365,    29,  3530,    24,  9032, 24568,  3789,  3671,   348,
             4,  2828,  6565,     3, 16250,     6,  1244, 15216,   350,  2676,
          2104,  1539,    53,  4520,    29,     3, 17617,  3530,     6,  1810,
           317,   663,  1132,     6,     8, 26495,   296,   382,   606,    65,
          1409,    11, 13615,     6,  4805,     7, 14886, 12339, 19438, 10148,
          6681,   998, 22160,     4, 26488,  2293,    19,    25, 19571,    24,
          7415,   105, 10856,  2293,    14,  4281,   947,     4,  2293,    14,
           872,   163,    16, 14645,  5362,   365,     6,    15,     3, 17617,
          6194,    30,  9213,   296,   314,   606,    65,  3023,    11,  3734,
          9213, 21812,    16,     3, 13976, 16211,  1466,  1223,  1832,  1184,
            53, 27745,  5007,  3850,   415,  3425,   903,   876,     4,  3862,
          5286,     3, 11054,    19,   211,   460,    98,  1026,    53,    17,
         12286,   231,   820, 26726, 16689,     3,  7180,  1770,   611,  9474,
          1720,     7,  2647,   368, 16395,     4,  5511, 18202,    22,  6434,
          1293,    16,     3,  7180, 10595,    28,  1395,   193, 24236,    29,
           946,     4, 13704,   428,  4281, 19358,  1728,     3,   892,    24,
           148,  7620,    51, 15149,     7,  8572,     8,  7944,     4,  4118,
         24654, 11316,     8, 13726,    13,  1433,   903,   876,   847,     4,
         15001,  2157,     7, 11580, 16768,  9586,    10,  2166,   242,  2752,
          1048,     1]], dtype=torch.int32)

model.eval()

with torch.no_grad():
    outputs = model.generate(
        input_ids
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False) 
    print(text)

