"""Compare asserted historical parameter budgets with executable configurations."""
import argparse
import json
from pathlib import Path
from ansg.config import ModelConfig
from ansg.model import ANSG,count_parameters

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/smoke.json')
    args=p.parse_args()
    cfg=ModelConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8')))
    m=ANSG(cfg)
    actual=count_parameters(m)
    reported={'chapter_4_trainable':18_000_000,'chapter_5_total':847_000_000,
              'chapter_5_base':774_000_000,'chapter_5_cultural':73_000_000,
              'chapter_5_lora_only':2_500_000}
    print(json.dumps({'actual_current_config':actual,'thesis_reported_reference_only':reported,
                      'matches_thesis':None,'note':'Model scales and architectures differ; arithmetic comparison is not proof.'},indent=2))
if __name__=='__main__':main()
