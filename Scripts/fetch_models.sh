#!/bin/sh
# Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
# Author: Torin Etheridge · 2026-10-09
# Downloads the validation model (SmolLM2-135M-Instruct, Apache-2.0) into ./Models.
#   GGUF files: bartowski/SmolLM2-135M-Instruct-GGUF   (~270 MB f16, ~145 MB Q8_0, ~92 MB Q4_0)
#   HF files  : HuggingFaceTB/SmolLM2-135M-Instruct     (~270 MB safetensors; only needed to regenerate references)
set -eu
cd "$(dirname "$0")/.."
mkdir -p Models/gguf Models/hf
B=https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main
H=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main
for f in SmolLM2-135M-Instruct-f16.gguf SmolLM2-135M-Instruct-Q8_0.gguf SmolLM2-135M-Instruct-Q4_0.gguf; do
  [ -f "Models/gguf/$f" ] || curl -fL -o "Models/gguf/$f" "$B/$f"
done
if [ "${1:-}" = "--with-reference" ]; then
  for f in model.safetensors tokenizer.json tokenizer_config.json config.json generation_config.json special_tokens_map.json vocab.json merges.txt; do
    [ -f "Models/hf/$f" ] || curl -fL -o "Models/hf/$f" "$H/$f"
  done
fi
shasum -a 256 Models/gguf/*.gguf
