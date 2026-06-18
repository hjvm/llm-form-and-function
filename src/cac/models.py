"""Model loading for CAC scoring — architecture dispatch + checkpoint support."""
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from cac.scoring import _seq2seq_sentinel_tokens, get_default_device

_LOADERS = {
    'mlm': AutoModelForMaskedLM,
    'ar': AutoModelForCausalLM,
    'seq2seq': AutoModelForSeq2SeqLM,
}


def load_model(model_config: Dict[str, str], device: Optional[torch.device] = None,
               revision: Optional[str] = None):
    """Load a model + tokenizer for CAC scoring.

    Args:
        model_config: ``{"name": repo_id, "type": "mlm"|"ar"|"seq2seq"}``.
        device: inference device (defaults to MPS->CUDA->CPU).
        revision: optional HF revision / checkpoint (e.g. a training-step tag).
            ``None`` loads the default revision, reproducing prior behavior exactly.

    Returns:
        ``(model, tokenizer)``.
    """
    if device is None:
        device = get_default_device()
        if device.type == "mps":
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

    model_name = model_config['name']
    model_type = model_config['type']
    if model_type not in _LOADERS:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of: mlm, ar, seq2seq")

    print(f"Loading {model_type.upper()} model: {model_name}" + (f" @ {revision}" if revision else ""))
    try:
        model = _LOADERS[model_type].from_pretrained(
            model_name, revision=revision, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model_type == 'seq2seq':
            _seq2seq_sentinel_tokens(tokenizer)  # fail fast on unsupported tokenizers
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise
