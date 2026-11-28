"""Toy check (CPU, ~1 min): can CSAM's H separate in-span from random pairs when motif spans
move around? Compares the two readings of pos_j in Eq. 4.13 (D7).

Usage: PYTHONPATH=src python scripts/csam_learnability_check.py
"""
import random, sys, torch
sys.path.insert(0, "scripts")
import train_phase1 as tp
from ansg.backbone.gpt2_bias import GPT2Spec, GPT2WithAttentionBias
from ansg.csam_bias import CSAMSettings, CulturalSchemaBias

def run(mode, steps=300, seed=0):
    torch.manual_seed(seed); rng = random.Random(seed)
    tp.CSAM_POS["mode"] = mode
    T, V, D = 48, 60, 48
    model = GPT2WithAttentionBias(GPT2Spec(vocab_size=V, n_positions=64, n_embd=D, n_layer=1, n_head=12))
    for p in model.parameters(): p.requires_grad_(False)
    csam = CulturalSchemaBias(CSAMSettings(n_head=12, d_model=D, graph_dim=8, layers=(0,)), torch.randn(2, 8))
    with torch.no_grad():
        for prm in (csam.v_pos_j, csam.v_pos_k, csam.v_schema): prm.normal_(0, 0.1)
    opt = torch.optim.AdamW(csam.parameters(), lr=2e-3)
    motif = list(range(50, 60))                       # motif spans use these tokens
    def batch(B=8):
        ids, spans = [], []
        for _ in range(B):
            x = [rng.randrange(0, 50) for _ in range(T)]
            a = rng.randrange(0, T - 8); L = rng.randrange(4, 8)
            x[a:a + L] = [rng.choice(motif) for _ in range(L)]
            ids.append(x); spans.append([(a, a + L)])
        return torch.tensor(ids), spans
    w = torch.zeros(8, 2); w[:, 0] = 1.0
    hist = []
    for step in range(steps):
        ids, spans = batch()
        pos, neg = tp.motif_pairs(spans, [T] * 8, [True] * 8, rng)
        loss = tp.cultural_loss(model, csam, ids, w, pos, neg)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
    return sum(hist[:20]) / 20, sum(hist[-20:]) / 20

for mode in ("position", "token_position"):
    print(mode, "L_cultural first/last 20 steps: %.3f -> %.3f (chance 1.386)" % run(mode))
