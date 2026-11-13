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
    """Weighted aggregate ONLY; inputs must be separately validated 0..1 scores.

    Weights from Ch.5 §5.1: 0.35,0.25,0.25,0.15. This does not implement
    the complete thesis annotation and expert assessment protocol.
    """
    for v in (rhetoric,dialect,structure,moral):
        if not 0<=v<=1:
            raise ValueError('CMPI components must be normalized to [0,1]')
    return .35*rhetoric+.25*dialect+.25*structure+.15*moral

def summarize(values):
    if not values: raise ValueError('Values required')
    return {'n':len(values),'mean':statistics.mean(values), 'sd':statistics.stdev(values) if len(values)>1 else 0.0}
