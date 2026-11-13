"""Schema-annotated training JSONL adapter and portable UTF-8 byte tokenizer.

Byte tokenizer is for smoke tests / low-resource runs; it does not replace the
thesis' 32k root-aware SentencePiece model.
"""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from ..config import DIALECTS

# Issue #11: a missing dialect is UNKNOWN, never MSA (id 0). DAL treats -1 rows
# with its learned soft gate. Missing schema ids mean "no active schema" (empty,
# fully masked), never a real cultural node; id 0 is PAD.
UNKNOWN_DIALECT_ID = -1


def _dialect_id(row, line_no):
    value = row.get('dialect_id')
    if value is None:
        return UNKNOWN_DIALECT_ID
    if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value < len(DIALECTS):
        raise ValueError(f'line {line_no}: dialect_id must be null, -1 or 0..{len(DIALECTS) - 1}, got {value!r}')
    return value


def _schema_ids(row, line_no, default_schema):
    value = row.get('schema_ids')
    if value is None:
        return [default_schema] if default_schema else []
    if not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in value):
        raise ValueError(f'line {line_no}: schema_ids must be a list of non-negative integers')
    return value

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
            self.rows.append({'tokens':tokens,'schema_ids':_schema_ids(row,line_no,default_schema),
                              'dialect_id':_dialect_id(row,line_no)})
        if not self.rows:
            raise ValueError('No usable examples with at least two tokens')

    def __len__(self): return len(self.rows)
    def __getitem__(self,i):return self.rows[i]

def collate(batch):
    # Pad inputs and labels separately; masked target tokens ignored in CE.
    max_t=max(len(r['tokens']) for r in batch)
    max_s=max(1, *(len(r['schema_ids']) for r in batch))
    ids=torch.zeros(len(batch),max_t,dtype=torch.long)
    labels=torch.full((len(batch),max_t),-100,dtype=torch.long)
    schemas=torch.zeros(len(batch),max_s,dtype=torch.long)
    schema_mask=torch.zeros(len(batch),max_s,dtype=torch.bool)
    dialects=torch.full((len(batch),),UNKNOWN_DIALECT_ID,dtype=torch.long)
    for i,r in enumerate(batch):
        n=len(r['tokens'])
        ids[i,:n]=torch.tensor(r['tokens'])
        labels[i,:n]=torch.tensor(r['tokens'])
        schemas[i,:len(r['schema_ids'])]=torch.tensor(r['schema_ids'])
        schema_mask[i,:len(r['schema_ids'])]=schemas[i,:len(r['schema_ids'])].ne(0)
        dialects[i]=UNKNOWN_DIALECT_ID if r.get('dialect_id') is None else r['dialect_id']
    return {'input_ids':ids,'labels':labels,'schema_ids':schemas,'schema_mask':schema_mask,'dialect_ids':dialects}
