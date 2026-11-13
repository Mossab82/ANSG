"""Honest evaluation helpers. Thesis-defined CMPI/ALDA require validated scorers."""
import math
import statistics

def bleu4(predictions,references):
    try:
        from sacrebleu import corpus_bleu
    except ImportError as exc:
        raise RuntimeError('Install evaluation extras: pip install -e .[metrics]') from exc
    if len(predictions)!=len(references):
        raise ValueError('Prediction/reference size mismatch')
    return corpus_bleu(predictions,[references]).score

def weighted_cmpi(rhetoric,dialect,structure,moral):
    """SURROGATE weighted aggregate — NOT the dissertation's CMPI.

    Computes 0.35*rhetoric + 0.25*dialect + 0.25*structure + 0.15*moral using the
    weights stated in Ch.5 §5.1. The dissertation gives no equation for CMPI and
    defines it inconsistently across chapters (docs/SPECIFICATION_CONFLICTS.md,
    C-EVAL-2); the four sub-scores have no definitions either. The CRAFT
    registry (ansg.evaluation.craft) therefore keeps CMPI 'incomplete'. Never
    report this value as CMPI or as a dissertation result.
    """
    for v in (rhetoric,dialect,structure,moral):
        if not 0<=v<=1:
            raise ValueError('CMPI components must be normalized to [0,1]')
    return .35*rhetoric+.25*dialect+.25*structure+.15*moral

def summarize(values):
    if not values: raise ValueError('Values required')
    return {'n':len(values),'mean':statistics.mean(values), 'sd':statistics.stdev(values) if len(values)>1 else 0.0}


# Machine-readable status so reports and tests can refuse to present this as CMPI.
weighted_cmpi.is_dissertation_cmpi = False
weighted_cmpi.metric_status = "surrogate"
weighted_cmpi.thesis_reference = "Ch.5 §5.1 weights only; CMPI undefined (C-EVAL-2)"
