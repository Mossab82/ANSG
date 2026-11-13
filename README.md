# ANSG — executable research reconstruction

**Real, testable PyTorch modules** for a staged reconstruction of *Automated Arabic Narrative Story Generation* based on the dissertation. This implementation is **new**; it is not claimed to reproduce historical source code, weights, or results.

## Implemented

- Decoder-only causal transformer with token/position embeddings and tied LM head.
- Cultural Schema Attention Mechanism with learned schema embeddings, cross-attention, learned cultural logit biases and 12 heads grouped 4/4/4 (small config).
- Dialectal Adaptation Layers using five rank-r adapters and schema-conditioned soft gates.
- CKG JSON loader, relational validation and simple lexical activation baseline.
- Supervised cross-entropy training, discriminator training, sequence-level policy-gradient adversarial refinement, and a clipped-PPO update function.
- Demonstration data loader, byte tokenizer, BLEU-4 optional evaluation, weighted CMPI aggregator, parameter audit and immutable-by-convention run manifest.
- Automated unit/integration tests and runnable supervised training entrypoint.

## Run a complete CPU smoke test

```bash
python -m pip install -e '.[test,metrics]'
pytest -q
python train.py --config configs/smoke.json --data examples/demo.jsonl --epochs 1 --output runs/smoke
```

The generated checkpoint and manifest record only the **new** experiment. The demo JSONL texts are synthetic and **not** the ANHC corpus. The demo's UTF-8 byte tokenizer is not the thesis' 32k root-aware tokenizer.

## Critical scope distinctions

This code implements functional, end-to-end *small-scale* ANSG components. The full-scale experimental replication is **not complete** because the original ANHC corpus and ontology are not supplied, pretrained AraGPT2 weights have not been integrated, several thesis descriptions are contradictory, and CRAFT/CMPI/ALDA metrics need the original measurement protocols.

The Appendix A.25 CSAM excerpt specifies a hidden width of 1280 and 12 heads, where `1280 / 12` is non-integral. We reject that combination rather than silently truncating dimensions. The code's default hidden size is 96 for a genuinely executable test. A projection to a compatible internal attention dimension remains a possible *documented design choice*, not an asserted historical implementation.

`configs/thesis_*.json` faithfully records competing **reported** values and must not be loaded as executable resolved configurations. See `docs/thesis_traceability.md` for detailed alignment and remaining work.

## Privacy and security

Legacy code on the `master` branch has hard-coded credentials. These should be immediately revoked/rotated and removed from Git history by the account holder; neither copying nor using them is necessary for this reconstruction. Do not publish private or copyrighted corpus material.
