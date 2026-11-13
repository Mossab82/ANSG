"""Executable 3-stage CPU/GPU research pipeline: CE -> adversarial -> PPO.

Real gradient updates and checkpointing; the final reward is a transparent
surrogate score, NOT CRAFT or original ANHC evaluation. Its purpose is to verify
stage orchestration before thesis-scale data, reward scorers and checkpoints.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from ansg.config import ModelConfig
from ansg.data import NarrativeDataset,collate
from ansg.model import ANSG,count_parameters
from ansg.training import (supervised_step,NarrativeDiscriminator,
                           discriminator_step,generator_adversarial_step,ppo_update)
from ansg.provenance import write_run_manifest,sha256_file


def surrogate_scores(samples,disc):
    """Simple, explicitly non-CRAFT smoke reward in [0,1]."""
    with torch.no_grad():
        realism=torch.sigmoid(disc(samples)).clamp(0,1)
        novelty=(samples[:,1:]!=samples[:,:-1]).float().mean(dim=-1)
        return (0.7*realism+0.3*novelty).clamp(0,1)


def run_pipeline(config_path,data_path,output_dir,steps_per_stage=1,seed=42,learning_rate=1e-4):
    if steps_per_stage<1:
        raise ValueError('steps_per_stage must be positive')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    config=ModelConfig.from_dict(json.loads(Path(config_path).read_text(encoding='utf-8')))
    dataset=NarrativeDataset(data_path,context_length=config.context_length)
    batches=DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator=ANSG(config).to(device)
    discriminator=NarrativeDiscriminator(config.vocab_size,config.hidden_size).to(device)
    gen_optimizer=torch.optim.AdamW(generator.parameters(),lr=learning_rate)
    disc_optimizer=torch.optim.AdamW(discriminator.parameters(),lr=learning_rate)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    data_label='synthetic' if Path(data_path).as_posix().endswith('examples/demo.jsonl') else 'unspecified'
    write_run_manifest(output/'manifest.json',config,data_path,seed,
                       extra={'data_label':data_label,'reward_type':'SURROGATE_NOT_CRAFT'})
    results=[]
    # Real supervised loss and optimizer updates.
    for index,batch in enumerate(batches):
        if index>=steps_per_stage:break
        batch={k:v.to(device) for k,v in batch.items()}
        loss=supervised_step(generator,batch,gen_optimizer)
        results.append({'stage':'supervised','step':index+1,'loss':loss})
    # Real discriminator + generator score-function updates.
    for index,batch in enumerate(batches):
        if index>=steps_per_stage:break
        batch={k:v.to(device) for k,v in batch.items()}
        prompt=batch['input_ids'][:,:8]
        schema=batch['schema_ids']
        dialect=batch['dialect_ids']
        fake=generator.generate(prompt,schema,dialect,max_new_tokens=4)
        dloss=discriminator_step(discriminator,batch['input_ids'][:,:12],fake[:,:12],disc_optimizer)
        gloss=generator_adversarial_step(generator,discriminator,prompt,schema,gen_optimizer,dialect,new_tokens=4)
        results.append({'stage':'adversarial','step':index+1,'discriminator_loss':dloss,'generator_loss':gloss})
    # Real on-policy sampling and clipped PPO updates with new-run surrogate reward.
    for index,batch in enumerate(batches):
        if index>=steps_per_stage:break
        batch={k:v.to(device) for k,v in batch.items()}
        prompt=batch['input_ids'][:,:8]
        schema=batch['schema_ids']
        dialect=batch['dialect_ids']
        samples=generator.generate(prompt,schema,dialect,max_new_tokens=4)
        states=samples[:,:-1]
        actions=samples[:,1:]
        with torch.no_grad():
            generator.eval()
            old=generator(states,schema,dialect)
            old_lp=F.log_softmax(old['logits'],dim=-1).gather(-1,actions.unsqueeze(-1)).squeeze(-1)
            score=surrogate_scores(samples,discriminator)
            # Credit only the 4 generated tokens, not the initial prompt.
            mask=torch.zeros_like(old_lp)
            mask[:,-4:]=1
            advantages=(score-.5)[:,None].expand_as(old_lp).clone()
            returns=old['values']+advantages
        metrics=ppo_update(generator,old_lp,states,schema,actions,advantages,returns,gen_optimizer,dialect,action_mask=mask)
        results.append({'stage':'ppo','step':index+1,'surrogate_reward':float(score.mean()),**metrics})
    torch.save({'model':generator.state_dict(),'config':config.__dict__,'stages_completed':['supervised','adversarial','ppo']},output/'model.pt')
    torch.save({'model':discriminator.state_dict()},output/'discriminator.pt')
    results_path=output/'training_results.json'
    results_path.write_text(json.dumps({'parameters':count_parameters(generator),
       'reconstruction':True,'reward_type':'SURROGATE_NOT_CRAFT','stages':results,
       'model_sha256':sha256_file(output/'model.pt')},indent=2),encoding='utf-8')
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/smoke.json')
    ap.add_argument('--data',default='examples/demo.jsonl')
    ap.add_argument('--output',default='runs/pipeline')
    ap.add_argument('--steps-per-stage',type=int,default=1)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    result=run_pipeline(args.config,args.data,args.output,args.steps_per_stage,args.seed)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
