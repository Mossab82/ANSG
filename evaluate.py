"""Evaluate a NEW model/checkpoint with transparent, labeled metrics.

No automatic assertion of the thesis' claimed CMPI or ALDA values.
"""
import argparse
import json
from pathlib import Path
import torch
from ansg.config import ModelConfig
from ansg.model import ANSG,count_parameters
from ansg.data import ByteTokenizer,NarrativeDataset
from ansg.provenance import sha256_file

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--test-data',required=True)
    parser.add_argument('--output',default='runs/evaluation.json')
    parser.add_argument('--max-new-tokens',type=int,default=32)
    args=parser.parse_args()
    try:
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    except TypeError:
        checkpoint=torch.load(args.checkpoint,map_location='cpu')
    config=ModelConfig.from_dict(checkpoint['config'])
    model=ANSG(config)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    dataset=NarrativeDataset(args.test_data,context_length=config.context_length)
    tokenizer=ByteTokenizer()
    results=[]
    with torch.no_grad():
        for i,row in enumerate(dataset.rows):
            prompt=torch.tensor([row['tokens'][:min(16,len(row['tokens'])-1)]],dtype=torch.long)
            schema=torch.tensor([row['schema_ids']],dtype=torch.long)
            generated=model.generate(prompt,schema,torch.tensor([row['dialect_id']]),max_new_tokens=args.max_new_tokens)
            results.append({'example':i,'prompt':tokenizer.decode(prompt[0].tolist()),
                'generated':tokenizer.decode(generated[0].tolist()),'schema_ids':row['schema_ids'],
                'dialect_id':row['dialect_id']})
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    output={'provenance':{'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha256_file(args.checkpoint),
         'data':str(args.test_data),'data_sha256':sha256_file(args.test_data)},
         'parameter_counts':count_parameters(model),
         'note':'Generated examples only. Does not implement CRAFT/ALDA or claim a dissertation table result.',
         'examples':results}
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(dest),'n_examples':len(results)},indent=2))
if __name__=='__main__':main()
