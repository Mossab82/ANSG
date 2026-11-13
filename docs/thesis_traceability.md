# Thesis traceability and open replication questions

| Implemented file | Thesis requirement | What is reproduced / what remains unresolved |
| --- | --- | --- |
| `src/ansg/csam.py` | §4.3, Eq.4.9–4.13; Appendix A.25 | Real learned attention and cultural bias. The precise pairwise `H_i[j,k]` relevance function has not been recovered; current schema-key bias is a documented variant. |
| `src/ansg/dal.py` | §4.4; A.3 | Five rank-8 LoRA-like dialect adaptations and schema-conditioned gating; target projection placement in pretrained backbone remains unresolved. |
| `src/ansg/knowledge.py` | §3.7; §4.3; A.4 | JSON graph + lexical baseline, not the reported OWL/CST-Onto inference or OWL2Vec* alignment. |
| `src/ansg/model.py` | §4.1 and §4.6 | Independently trainable transformer with integrated CSAM/DAL. Not a substitute for actual AraGPT2 checkpoint. |
| `src/ansg/training.py` | §§4.5, 5.1–5.6 | Supervised CE, discriminator, REINFORCE and clipped PPO update; full three-phase orchestration and CRAFT reward scorers remain. |
| `src/ansg/evaluation.py` | Chapter 6 | BLEU-4 via SacreBLEU and explicitly documented CMPI weighted aggregate, not complete expert-based evaluation protocol. |
| `src/ansg/data.py` | Chapter 3; §5.4 | Runnable JSONL and byte-level smoke tokenizer; replace with root-aware 32k SentencePiece for thesis replication. |
| `src/ansg/provenance.py` | §5.2; Appendix A | New experiment SHA256 and actual run time, no fabricated historical hashes. |

## Reported contradictions (must resolve through research)

1. §4.1.6 355M GPT-2 Medium backbone vs Table 5.1 774M backbone vs A.3 `aubmindlab/aragpt2-large`.
2. §4.1.6 18M trainable cultural components vs Table 5.1 73M cultural additions vs §8.7.1 73M for CSAM alone.
3. §4.6 LoRA 737,280 vs §4.1.6 1.2M vs Table 5.1 2.5M LoRA-only.
4. A.25 `hidden_size=1280`, `cultural_heads=12`, `head_dim=hidden_size//num_heads`: incompatible without further dimension mapping.
5. §5.1 30+35+35 epoch training vs A.3 phase1 10 epochs vs A.26 four-phase training.
6. §5.1 three-term reward 0.45/0.35/0.20 vs A.3 four-term 0.35/0.30/0.25/0.10.
7. §5.2 A100 40GB, PyTorch 2.0.1 vs A.2 A100 80GB, PyTorch 1.12.0.
8. §4.6 `898×768` schema parameters vs A.25 `1847×256` embedding spec.

## Roadmap to full research reproduction

1. Secure legacy history and preserve old commits.
2. Verify licenses, original ANHC annotations and ontology file provenance.
3. Establish exact pretrained backbone, tokenizer and CSAM head mapping with actual tensor shapes.
4. Implement Eq.4.12–4.13 pairwise relevance and alignment loss Eq.4.18 after clarifying dimensions.
5. Build root-aware tokenizer, distributed supervised / adversarial / PPO pipelines and verified CRAFT rewards.
6. Run ablations, five-seed evaluations and compare against dissertation tables with confidence intervals.
7. Publish code, model cards, data statements and run logs with truthful execution timestamps.

## Pretrained backbone integration

`src/ansg/pretrained.py` provides an adapter around a Hugging Face causal LM with hidden-state cultural cross-attention and DAL. It uses an explicit 1280→1296→1280 cultural projection to address head divisibility when necessary. **This bridge does not inject pairwise biases into the pretrained LM's internal attention logits**; claim of strict architectural equivalence requires further integration.

The from-scratch `ANSG` model does implement *pairwise* culture-conditioned logit biases through `PairwiseCulturalBias`, projecting twelve cultural bias heads onto the actual decoder attention heads. That projection is a novel engineering decision and not established by the thesis.
