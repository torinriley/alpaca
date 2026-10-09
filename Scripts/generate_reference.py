#!/usr/bin/env python3
# Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
# Author: Torin Etheridge · 2026-10-09
"""Real-model references (development-time only; alpaca never calls Python at runtime).

Inputs  : Models/hf (SmolLM2-135M-Instruct, Hugging Face format) and Models/gguf/*.gguf
Outputs : Tests/AlpacaTokenizerTests/Fixtures/corpus.json        tokenizer compatibility corpus (HF token ids)
          Tests/AlpacaIntegrationTests/Fixtures/reference.json   last-position logits + greedy continuations

For each GGUF file the reference model is Hugging Face `LlamaForCausalLM` loaded from *that file's weights*
(dequantised to float32 by the `gguf` package) and run in float32 on CPU. This isolates alpaca's arithmetic
from quantisation error: the quantised reference sees exactly the weights alpaca sees. The original
bf16 safetensors checkpoint is included as "hf_original" to measure total deviation from the source model.
"""
import base64, json, pathlib, random, numpy as np, torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Canonical `tokenizers` library, read straight from tokenizer.json. (transformers 5.x re-assembles the
# pre-tokenizer for Llama-type tokenizers and diverges from the original on digits/whitespace, so it is not used.)
_tok = Tokenizer.from_file(str(ROOT / "Models/hf/tokenizer.json"))
class tok:
    encode = staticmethod(lambda t, add_special_tokens=False: _tok.encode(t, add_special_tokens=False).ids)
    decode = staticmethod(lambda ids: _tok.decode(list(ids) if not hasattr(ids, "item") else [int(ids)], skip_special_tokens=False))

# ---------- tokenizer corpus ----------
texts = [
    "", " ", "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "  leading spaces", "trailing spaces   ", "multiple   inner    spaces", "tab\tseparated\tvalues", "line1\nline2\n\nline4",
    "\r\nwindows line endings\r\n", " \n \n ", "Punctuation: (a), [b], {c}; \"quoted\" 'single' -- dash... ellipsis!?",
    "it's they're we've I'm you'll he'd THEY'RE", "3.14159", "1234567890", "In 2024, 42 apples cost $1,234.56 (12%).",
    "x² + y³ = ½ and ٣٤٥ and Ⅷ", "café naïve façade Zoë", "日本語のテキストです。", "中文文本测试，包含标点。", "العربية: مرحبا بالعالم",
    "Привет, мир! Это тест.", "한국어 텍스트", "हिन्दी पाठ", "emoji 👍 🚀 🧑‍🚀 🇺🇸 ❤️", "mixed: Hello 世界 🌍 Привет 123",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "<|endoftext|>", "<|im_start", "<|im_end|><|im_end|>",
    "def f(x):\n    return x**2  # square\n", "https://example.com/path?query=1&b=2#frag", "a" * 200, "word " * 50,
    " non-breaking space", "zero​width", "combining é vs é", "\x00null", "ends with newline\n", "\n\n\n\n", "       ",
]
rnd = random.Random(7)
alphabet = list(" \n\tabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"!?()-_é日本語🙂Ω٣²") + ["<|im_start|>", "<|endoftext|>"]
for _ in range(150):
    texts.append("".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 60))))
corpus = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in texts]
p = ROOT / "Tests/AlpacaTokenizerTests/Fixtures/corpus.json"; p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(corpus, ensure_ascii=False)); print("corpus", len(corpus), "->", p)

# ---------- model references ----------
prompts = [
    "The capital of France is",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    ("Transformers process sequences with attention. Each layer mixes information between tokens, "
     "then applies a feed-forward network to every position independently. Rotary embeddings encode position "
     "by rotating pairs of query and key channels by an angle proportional to the token index. "
     "In 2017, the original architecture was introduced; since then, decoder-only variants have dominated. "
     "To generate text, the model repeatedly predicts the next token and appends it to the context."),
]
GEN = 24
def run(model, label):
    out = []
    for text in prompts:
        ids = tok.encode(text, add_special_tokens=False)
        with torch.no_grad():
            last = model(torch.tensor([ids])).logits[0, -1].float()
            seq, margins = list(ids), []
            for _ in range(GEN):
                lg = model(torch.tensor([seq])).logits[0, -1]
                top2 = torch.topk(lg, 2)
                margins.append((top2.values[0] - top2.values[1]).item())
                seq.append(int(top2.indices[0]))
        out.append(dict(prompt=text, ids=ids, last_logits=base64.b64encode(last.numpy().astype("<f4").tobytes()).decode(),
                        generated=seq[len(ids):], margins=margins))
        print(label, repr(text[:30]), "->", repr(tok.decode(seq[len(ids):])[:60]))
    return out

ref = {}
ref["hf_original"] = run(LlamaForCausalLM.from_pretrained(ROOT / "Models/hf", dtype=torch.float32).eval(), "hf_original")
for name in ["f16", "Q8_0", "Q4_0"]:
    m = LlamaForCausalLM.from_pretrained(ROOT / "Models/gguf", gguf_file=f"SmolLM2-135M-Instruct-{name}.gguf",
                                         dtype=torch.float32, device_map="cpu").eval()
    ref[name] = run(m, name)
p = ROOT / "Tests/AlpacaIntegrationTests/Fixtures/reference.json"; p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(ref)); print("reference ->", p, p.stat().st_size)
