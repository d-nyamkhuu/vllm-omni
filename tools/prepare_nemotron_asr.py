# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Repackage the original .nemo tensors for vLLM's safetensors loader."""

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from transformers import LlamaTokenizer


def prepare(archive, output, *, packed_streaming=True):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        candidate = Path(staging) / "model"
        candidate.mkdir()
        _convert(archive, candidate, packed_streaming)
        candidate.rename(output)


def _convert(archive, output, packed_streaming):
    with tempfile.TemporaryDirectory(dir=output) as tmp, tarfile.open(archive) as tar:
        members = {Path(member.name).name: member for member in tar if member.isfile()}
        cfg = yaml.safe_load(tar.extractfile(members["model_config.yaml"]))
        if cfg["target"].split(".")[-1] != "EncDecRNNTBPEModelWithPrompt":
            raise ValueError("Expected the language-prompted RNNT checkpoint")
        for key, dest in (
            ("model_path", "tokenizer.model"),
            ("vocab_path", "vocab.txt"),
            ("spe_tokenizer_vocab", "tokenizer.vocab"),
        ):
            with (output / dest).open("wb") as f:
                shutil.copyfileobj(tar.extractfile(members[cfg["tokenizer"][key].removeprefix("nemo:")]), f)
            cfg["tokenizer"][key] = dest
        weights = Path(tmp) / "model_weights.ckpt"
        with weights.open("wb") as f:
            shutil.copyfileobj(tar.extractfile(members[weights.name]), f)
        tensors = torch.load(weights, map_location="cpu", weights_only=True)
        save_file({f"asr.{k}": v.contiguous() for k, v in tensors.items()}, output / "model.safetensors")
    tok = LlamaTokenizer.from_pretrained(
        output,
        local_files_only=True,
        legacy=True,
        bos_token=None,
        eos_token=None,
        add_bos_token=False,
        add_eos_token=False,
    )
    tok.save_pretrained(output)
    config = dict(
        model_type="nemotron_unified_asr",
        architectures=["NemotronUnifiedASR"],
        hidden_size=cfg["encoder"]["d_model"],
        num_attention_heads=cfg["encoder"]["n_heads"],
        num_hidden_layers=cfg["encoder"]["n_layers"],
        vocab_size=len(tok),
        max_position_embeddings=4096,
        asr_config=cfg,
        packed_streaming=packed_streaming,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", action="store_true", help="Use the released NeMo encoder and greedy decoder.")
    args = parser.parse_args()
    prepare(args.archive, args.output, packed_streaming=not args.reference)
