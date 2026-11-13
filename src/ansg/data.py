"""Schema-annotated training JSONL adapter and portable UTF-8 byte tokenizer.

Byte tokenizer is for smoke tests / low-resource runs; it does not replace the
thesis' 32k root-aware SentencePiece model.
"""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class ByteTokenizer:
    vocab_size = 256
    def encode(self,text):
        return list(text.encode('utf-8'))
    def decode(self,tokens):
        return bytes([int(x) for x in tokens]).decode('utf-8',errors='replace')

class NarrativeDataset(Dataset):
    def __init__(self,filename,context_length=128,default_schema=0):
        self.rows=[]
        tokenizer=ByteTokenizer()
        for line_no,line in enumerate(Path(filename).read_text(encoding='utf-8').splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            if not isinstance(row.get('text'),str) or not row['text']:
                raise ValueError(f'line {line_no}: text is required')
            tokens=tokenizer.encode(row['text'])[:context_length]
            if len(tokens)<2: continue
            schema=row.get('schema_ids',[default_schema])
            self.rows.append({'tokens':tokens,'schema_ids':schema,'dialect_id':int(row.get('dialect_id',0))})
        if not self.rows:
            raise ValueError('No usable examples with at least two tokens')

    def __len__(self): return len(self.rows)
    def __getitem__(self,i):return self.rows[i]

def collate(batch):
    # Pad inputs and labels separately; masked target tokens ignored in CE.
    max_t=max(len(r['tokens']) for r in batch)
    max_s=max(len(r['schema_ids']) for r in batch)
    ids=torch.zeros(len(batch),max_t,dtype=torch.long)
    labels=torch.full((len(batch),max_t),-100,dtype=torch.long)
    schemas=torch.zeros(len(batch),max_s,dtype=torch.long)
    dialects=torch.zeros(len(batch),dtype=torch.long)
    for i,r in enumerate(batch):
        n=len(r['tokens'])
        ids[i,:n]=torch.tensor(r['tokens'])
        labels[i,:n]=torch.tensor(r['tokens'])
        schemas[i,:len(r['schema_ids'])]=torch.tensor(r['schema_ids'])
        dialects[i]=r['dialect_id']
    return {'input_ids':ids,'labels':labels,'schema_ids':schemas,'dialect_ids':dialects}
