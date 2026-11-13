"""Real supervised training entry point. Other phases live in ansg.training API."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ansg.config import ModelConfig
from ansg.data import NarrativeDataset,collate
from ansg.model import ANSG,count_parameters
from ansg.training import supervised_step
from ansg.provenance import write_run_manifest,sha256_file

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/smoke.json')
    ap.add_argument('--data',default='examples/demo.jsonl')
    ap.add_argument('--output',default='runs/smoke')
    ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg=ModelConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8')))
    if cfg.vocab_size != 256:
        raise ValueError('This CLI expects demo ByteTokenizer (vocab_size=256); use a matching tokenizer for pretrained models')
    dataset=NarrativeDataset(args.data,context_length=cfg.context_length)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ANSG(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    write_run_manifest(output/'run_manifest.json',cfg,args.data,args.seed)
    print('parameters:',json.dumps(count_parameters(model)))
    for epoch in range(args.epochs):
        losses=[]
        for raw in loader:
            batch={k:v.to(device) for k,v in raw.items()}
            losses.append(supervised_step(model,batch,opt))
        print(f'epoch={epoch+1} mean_loss={sum(losses)/len(losses):.5f}')
    torch.save({'model':model.state_dict(),'config':cfg.__dict__,'epoch':args.epochs},output/'checkpoint.pt')
    print('checkpoint_sha256:',sha256_file(output/'checkpoint.pt'))

if __name__=='__main__': main()
