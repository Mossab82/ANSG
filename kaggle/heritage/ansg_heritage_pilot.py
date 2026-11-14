"""Reproducible Arabic heritage LoRA pilot; experimental, NOT a validated ANSG result.

This run adapts a pretrained backbone to *One Thousand and One Nights* text.
It does not train/verify the ANSG neuro-symbolic CSAM/DAL architecture or dialect.
"""
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OUTPUT = Path("/kaggle/working/ansg_heritage_pilot")
SOURCE = "https://ar.wikisource.org/w/api.php"
TITLE_PREFIX = "ألف ليلة وليلة (مؤسسة هنداوي، 2022)/الجزء الأول/فلما كانت الليلة "
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MAX_STEPS = int(os.environ.get("ANSG_MAX_STEPS", "140"))
MAX_LENGTH = 256
BATCH_ACCUM = 4
LR = 1.2e-4
TRAIN_NIGHTS = list(range(1, 15))
EVAL_NIGHTS = list(range(15, 19))
PROMPTS = {
    "msa": "اكتب بداية قصة عربية أصلية عن صياد وجد صندوقًا غامضًا ثم اكتشف صاحبه الحقيقي:",
    "levantine": "احكِ قصة قصيرة بلهجة أهل العقبة عن شاب لقى صندوق ورجع يدور على صاحبه:",
    "egyptian": "احكي حكاية قصيرة بالمصري عن صياد لقى أمانة وقرر يرجعها لصاحبها:",
    "moroccan": "حكي ليا حكاية قصيرة بالدارجة المغربية على واحد لقى أمانة وبغا يرجعها:",
}

def save_json(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT / name
    dest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Frozen, already-extracted text corpus from Github Actions #43561547592.
# The commit is immutable, preventing accidental replacement of the training set.
CORPUS_COMMIT = "da54c488c9e94d049b5ed38b28a3d8005aac88f7"
CORPUS_URL = ("https://raw.githubusercontent.com/Mossab82/ANSG/"
              + CORPUS_COMMIT + "/data/corpora/classical_arabic/stories.jsonl")
TRAIN_IDS = {"arabian_nights-001", "arabian_nights-002", "arabian_nights-003",
             "arabian_nights-004", "kalila_dimna-006", "kalila_dimna-007",
             "kalila_dimna-008", "kalila_dimna-009", "kalila_dimna-010",
             "kalila_dimna-011", "kalila_dimna-012", "kalila_dimna-013",
             "kalila_dimna-014", "kalila_dimna-017"}
VALIDATION_IDS = {"kalila_dimna-015", "kalila_dimna-016", "kalila_dimna-018"}
TEST_IDS = {"kalila_dimna-019", "kalila_dimna-020"}
EXCLUDED_IDS = {"arabian_nights-005"}  # Sinbad likely overlaps the 4 anthology parts.

def load_verified_corpus():
    """Load a frozen historical text corpus; verify bytes, IDs and page-level groups."""
    req=Request(CORPUS_URL,headers={
        "User-Agent":"ANSGResearchPilot/0.3 (https://github.com/Mossab82/ANSG)",
        "Accept":"text/plain"})
    with urlopen(req,timeout=90) as res:
        raw=res.read()
    rows=[json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    expected=TRAIN_IDS|VALIDATION_IDS|TEST_IDS|EXCLUDED_IDS
    if {r["id"] for r in rows}!=expected or len(rows)!=len(expected):
        raise ValueError("Source corpus ids changed or duplicates detected")
    if sum(r["word_count"] for r in rows)!=195197:
        raise ValueError("Corpus is not the reviewed 195197-word revision")
    for r in rows:
        if sha(r["text"])!=r["sha256"]:
            raise ValueError("Source SHA mismatch: "+r["id"])
    splits={"train":[r for r in rows if r["id"] in TRAIN_IDS],
            "validation":[r for r in rows if r["id"] in VALIDATION_IDS],
            "test":[r for r in rows if r["id"] in TEST_IDS]}
    if min(len(x) for x in splits.values())<2:
        raise ValueError("Insufficient distinct chapter/cycle groups")
    save_json("source_manifest.json",{
        "corpus_source":CORPUS_URL,
        "source_git_commit":CORPUS_COMMIT,
        "record_count":len(rows),
        "source_ids":{k:[r["id"] for r in v] for k,v in splits.items()},
        "excluded":sorted(EXCLUDED_IDS),
        "pages":[{k:v for k,v in r.items() if k!="text"} for r in rows],
        "download_utc":datetime.now(timezone.utc).isoformat(),
        "license_notice":"Only ancient narratives and Wikisource editions; verify exact page licenses and jurisdiction prior to reuse/publication",
        "research_scope":"Experimental heritage-text LoRA pilot only, NOT dissertation ANHC/CKG/CSAM/DAL."
    })
    return splits

def byte_arabic_ratio(s):
    letters=[c for c in s if c.isalpha()]
    if not letters:return 0.0
    return round(sum('\u0600'<=c<='\u06ff' for c in letters)/len(letters),4)

def repeated_ngrams(s,n=3):
    w=s.split()
    if len(w)<n:return 0.0
    grams=[tuple(w[i:i+n]) for i in range(len(w)-n+1)]
    return round(1.0-len(set(grams))/len(grams),4)

def chunks(rows,tokenizer):
    out=[]
    for row in rows:
        ids=tokenizer(row["text"],add_special_tokens=False)["input_ids"]
        for start in range(0,len(ids)-MAX_LENGTH+1,MAX_LENGTH):
            chunk=ids[start:start+MAX_LENGTH]
            if len(chunk)>=64:out.append({"ids":chunk,"night":row["night"]})
    return out

def generate_samples(model, tokenizer, stage):
    import torch
    samples=[]
    model.eval()
    for label,prompt in PROMPTS.items():
        tokens=tokenizer(prompt,return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated=model.generate(**tokens,max_new_tokens=130,
               do_sample=False,pad_token_id=tokenizer.eos_token_id)
        new=tokenizer.decode(generated[0,tokens["input_ids"].shape[1]:],
                             skip_special_tokens=True)
        samples.append({"prompt_id":label,"prompt":prompt,"generated":new,
                        "arabic_letter_ratio":byte_arabic_ratio(new),
                        "repeated_trigram_fraction":repeated_ngrams(new),
                        "words":len(new.split()),"stage":stage})
    save_json(stage+"_samples.json",samples)
    return samples

def main():
    # Kaggle notebooks may lack PEFT and MediaWiki parsing packages by default.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "peft==0.15.2", "transformers==4.51.3", "accelerate==1.7.0",
        "mwparserfromhell==0.7.2", "beautifulsoup4==4.13.4"], check=True, timeout=300)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    OUTPUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required: do not train this pilot on a CPU")
    device_name=torch.cuda.get_device_name(0)
    if MAX_STEPS<1 or MAX_STEPS>300:
        raise ValueError("ANSG_MAX_STEPS must be 1..300")
    splits=load_verified_corpus()
    train_src=splits["train"]
    eval_src=splits["validation"]
    test_src=splits["test"]
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    train=chunks(train_src,tokenizer)
    heldout=chunks(eval_src,tokenizer)
    test_chunks=chunks(test_src,tokenizer)
    if len(train)<30 or len(heldout)<3 or len(test_chunks)<2:
        raise RuntimeError("Not enough train and later-night diagnostic chunks")
    save_json("data_report.json",{
        "train_chunks":len(train),"heldout_chunks":len(heldout),
        "train_source_ids":sorted(TRAIN_IDS),"validation_source_ids":sorted(VALIDATION_IDS),
        "test_source_ids":sorted(TEST_IDS),"excluded_source_ids":sorted(EXCLUDED_IDS),
        "tokenizer":MODEL,"context_tokens":MAX_LENGTH,
        "note":"Real Arabic historical text; chapters/cycles grouped by source, no gold event/dialect labels. Holdout genre differs from main Nights training distribution.",
        "split_integrity":"PAGE_CYCLE_GROUPED_NOT_FULLY_NESTED_STORY_AUDITED",
    })
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.float16,device_map={"":0},trust_remote_code=False)
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    before=generate_samples(model,tokenizer,"baseline")
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",
                    task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"])
    model=get_peft_model(model,lora)
    model.enable_input_require_grads()
    model.train()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=LR)
    history=[]
    best=float("inf")
    no_improve=0

    def eval_loss():
        model.eval()
        losses=[]
        with torch.no_grad():
            for chunk in heldout[:min(24,len(heldout))]:
                ids=torch.tensor([chunk["ids"]],device="cuda")
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    loss=model(input_ids=ids,labels=ids).loss
                losses.append(float(loss))
        model.train()
        return sum(losses)/len(losses)

    baseline_loss=eval_loss()
    save_json("baseline_metrics.json",{"heldout_ce_loss":baseline_loss,
        "heldout_selection":"first 24 chunks of Kalila validation chapter groups",
        "evaluation_warning":"Grouped by source chapter/cycle; nested sub-story overlaps and shared frame formulae not independently audited. Domain differs from Arabian Nights.",
        "sample_prompts":list(PROMPTS),"performance_claim":"none"})
    rng=random.Random(SEED)
    started=time.monotonic()
    for step in range(1,MAX_STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        sample_losses=[]
        for _ in range(BATCH_ACCUM):
            chunk=rng.choice(train)
            ids=torch.tensor([chunk["ids"]],device="cuda")
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                loss=model(input_ids=ids,labels=ids).loss/BATCH_ACCUM
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            sample_losses.append(float(loss.detach())*BATCH_ACCUM)
        torch.nn.utils.clip_grad_norm_(parameters,1.0)
        optimizer.step()
        record={"step":step,"train_ce_loss":round(sum(sample_losses)/len(sample_losses),5)}
        if step==1 or step%20==0 or step==MAX_STEPS:
            score=eval_loss()
            record["heldout_ce_loss"]=round(score,5)
            print(json.dumps(record),flush=True)
            if score < best-0.005:
                best=score
                no_improve=0
            else:
                no_improve+=1
            if step%40==0 or step==MAX_STEPS:
                checkpoint=OUTPUT/("adapter_step_%03d"%step)
                model.save_pretrained(checkpoint,safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
                save_json("train_progress.json",{
                    "run_type":"heritage_pretrained_LoRA_pilot",
                    "status":"running","model":MODEL,
                    "steps_finished":step,"max_steps":MAX_STEPS,
                    "seed":SEED,"lr":LR,"lora_rank":8,
                    "single_gpu_used":device_name,
                    "history":history+[record]})
            if no_improve>=4:
                history.append(record)
                print("Early stopping: 4 evaluations without improvement.",flush=True)
                break
        history.append(record)

    model.config.use_cache=True
    after=generate_samples(model,tokenizer,"adapted")
    model.save_pretrained(OUTPUT/"adapter_final",safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT/"adapter_final")
    report={
        "status":"completed","experiment":"heritage_adapter_pilot",
        "model":MODEL,"base_model_is_pretrained":True,"adapter_type":"LoRA",
        "trained_neuro_symbolic_ANSG":False,
        "trained_CSAM_DAL":False,"trained_dialect_data":False,
        "dataset_kind":"real_heritage_text_no_gold_event_dialect_labels",
        "evaluation_split":"source-page/cycle grouped Kalila validation; separate untouched Kalila test; nested overlap not fully audited",
        "steps_finished":len(history),"minutes_elapsed":round((time.monotonic()-started)/60,2),
        "baseline_heldout_loss":round(baseline_loss,5),
        "best_heldout_loss":round(best,5),
        "last_heldout_loss":next((x["heldout_ce_loss"] for x in reversed(history) if "heldout_ce_loss" in x),None),
        "gpu":device_name,"train_chunks":len(train),"heldout_chunks":len(heldout),
        "test_chunks_reserved_not_used_for_selection":len(test_chunks),
        "source_git_commit":CORPUS_COMMIT,
        "quality_validation":"Not validated. Human evaluation of Arabic fluency, culture, dialect and plot required.",
        "automated_quality_checks":"Arabic letter ratio, trigram repetition only; NOT equivalent to CRAFT.",
        "sample_count_before":len(before),"sample_count_after":len(after),
        "history":history
    }
    save_json("final_report.json",report)
    print("COMPLETE:",json.dumps({k:report[k] for k in ("status","steps_finished",
        "baseline_heldout_loss","best_heldout_loss","last_heldout_loss")}),flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        save_json("failure_report.json",{
            "status":"failed","error_type":type(exc).__name__,"message":str(exc),
            "timestamp_utc":datetime.now(timezone.utc).isoformat()
        })
        raise
