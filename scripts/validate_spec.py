"""Statically flag a central inconsistency in the dissertation CSAM excerpt."""
import json
from pathlib import Path

def main():
    cfg=json.loads(Path('configs/thesis_appendix_a3_reported.json').read_text())
    h=cfg['hidden_size']
    c=cfg['cultural_heads']
    print(f'appendix hidden_size={h}, cultural_heads={c}')
    print(f'head_dim={h//c}, unused_dimensions={h%c}')
    print('REQUIRES DESIGN DECISION' if h%c else 'DIMENSIONS COMPATIBLE')
if __name__=='__main__':main()
