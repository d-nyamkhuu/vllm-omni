# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
import tarfile

import pytest
import sentencepiece as sp
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer

from tools.prepare_nemotron_asr import prepare

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_conversion_preserves_piece_ids_and_checkpoint_weights(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sp.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello streaming speech", "speech in small windows"] * 8),
        model_prefix=str(source / "tiny"),
        model_type="bpe",
        vocab_size=28,
        bos_id=-1,
        eos_id=-1,
    )
    (source / "vocab.txt").write_text("hello")
    config = {
        "target": "EncDecRNNTBPEModelWithPrompt",
        "encoder": {"d_model": 8, "n_heads": 1, "n_layers": 1},
        "tokenizer": {
            "model_path": "nemo:tiny.model",
            "vocab_path": "nemo:vocab.txt",
            "spe_tokenizer_vocab": "nemo:tiny.vocab",
        },
    }
    (source / "model_config.yaml").write_text(yaml.safe_dump(config))
    weight = torch.arange(6).reshape(2, 3)
    torch.save({"encoder.weight": weight}, source / "model_weights.ckpt")
    archive = tmp_path / "tiny.nemo"
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source, arcname=".")
    output = tmp_path / "prepared"
    prepare(archive, output)
    cfg = json.loads((output / "config.json").read_text())
    expected = sp.SentencePieceProcessor(model_file=str(source / "tiny.model"))
    tok = AutoTokenizer.from_pretrained(output)
    assert cfg["vocab_size"] == len(tok) == expected.vocab_size()
    text = "hello streaming speech"
    assert tok.encode(text, add_special_tokens=False) == expected.encode(text)
    torch.testing.assert_close(load_file(output / "model.safetensors")["asr.encoder.weight"], weight)
    with pytest.raises(FileExistsError):
        prepare(archive, output)
