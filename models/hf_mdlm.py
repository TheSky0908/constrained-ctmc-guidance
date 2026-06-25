"""Adapter exposing the pretrained HuggingFace MDLM (``kuleshov-group/mdlm-owt``)
through this repo's DIT backbone interface.

``mdlm-owt`` is the masked diffusion LM from Sahoo et al. (2024) trained on
OpenWebText with the GPT-2 tokenizer. Its absorbing ``[MASK]`` token sits at
index 50257, giving ``vocab_size = 50258`` and ``model_length = 1024``; it is
*unconditional* and uses ``time_conditioning = False``. These exactly match the
values this repo derives for a GPT-2 tokenizer (see ``Diffusion.__init__``:
GPT-2 has no mask token, so ``mask_index = 50257`` and ``vocab_size`` becomes
50258), so no vocab remapping is needed.

The pretrained model's ``forward`` signature is ``(input_ids, timesteps, ...)``
which does not match the repo's ``backbone(indices, sigma, cond=None,
x_emb=None, return_hidden_states=False)`` convention used in
``Diffusion.forward``. We therefore wrap the model's inner ``DITBackbone``
(``forward(indices, sigma, output_hidden_states)``), which IS interface-
compatible once we drop the unused ``cond`` / ``x_emb`` arguments.
"""
import torch
import torch.nn as nn
import transformers


class HFMDLMBackbone(nn.Module):
  """Wrap ``kuleshov-group/mdlm-owt`` as a repo-compatible DIT backbone.

  ``cond`` and ``x_emb`` are accepted (for signature compatibility with
  ``models.dit.DIT``) but must be ``None`` — the base model is unconditional and
  guidance is applied externally (D-CBG / adaptive-dual operate on the
  classifier, never on the base model's conditioning input).
  """

  def __init__(self, config):
    super().__init__()
    name = config.model.pretrained_model_name_or_path
    try:
      mdlm = transformers.AutoModelForMaskedLM.from_pretrained(
        name, trust_remote_code=True)
    except Exception:  # offline fallback: resolve the local snapshot dir
      from huggingface_hub import snapshot_download
      local_dir = snapshot_download(name, local_files_only=True)
      mdlm = transformers.AutoModelForMaskedLM.from_pretrained(
        local_dir, trust_remote_code=True)
    # Inner DITBackbone: forward(indices, sigma, output_hidden_states=False)
    # -> (logits, all_hidden_states).
    self.backbone = mdlm.backbone

  def forward(self, indices, sigma, cond=None, x_emb=None,
              return_hidden_states=False):
    if cond is not None:
      raise NotImplementedError(
        'HFMDLMBackbone is unconditional; `cond` must be None.')
    if x_emb is not None:
      raise NotImplementedError(
        'HFMDLMBackbone does not support `x_emb` (soft-embedding) input.')
    logits, hidden_states = self.backbone(
      indices, sigma, output_hidden_states=return_hidden_states)
    if return_hidden_states:
      return logits, hidden_states
    return logits
