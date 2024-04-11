#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import torch
from torchtext.models import T5Bundle
from torchtext.models.t5.bundler import GenerationUtilsForT5
from torchtext.prototype.generate import GenerationUtils

model_path = "/opt/model/t5-qiita-title-generation"

# TorchText形式のモデルを作成
model = T5Bundle.build_model_from_huggingface_ckpt(model_path)

# TorchTextのT5モデルフォーマットに変換
gen_kwargs = {}
generation_model = GenerationUtilsForT5(model, **gen_kwargs)
assert torch.jit.isinstance(generation_model, GenerationUtils)

# TorchScript形式に変換
scripted_model = torch.jit.script(generation_model)
torch.jit.save(scripted_model, "traced_jit.pt")
