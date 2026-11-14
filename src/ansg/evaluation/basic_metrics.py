"""Evaluation helpers. CMPI's equation is defined in the dissertation; its dimension scorers and ALDA are not."""
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
    """The dissertation's CMPI equation (kept under its historical name).

    Computes 0.35*rhetoric + 0.25*dialect + 0.25*structure + 0.15*moral, the
    AHP-weighted composite that defines CMPI in Ch.5 §5.1 (p.131) and §8.7.4
    (p.223); see :func:`ansg.evaluation.cmpi.ansg_cmpi`, which this delegates to.
    The dissertation does not specify how the four dimension scores are obtained
    from a text, so a value is only as meaningful as the caller's dimension
    scores; it never reproduces a dissertation result. CULTURA, CST-Onto and
    MM-GTA use different metrics of the same name
    (configs/evaluation/cmpi_variants.yaml).
    """
    from .cmpi import ansg_cmpi
    return ansg_cmpi(rhetoric,dialect,structure,moral)

def summarize(values):
    if not values: raise ValueError('Values required')
    return {'n':len(values),'mean':statistics.mean(values), 'sd':statistics.stdev(values) if len(values)>1 else 0.0}


# Machine-readable status so reports and tests can state exactly what this is.
weighted_cmpi.is_dissertation_cmpi = False            # historical flag: a computed value is not a dissertation result
weighted_cmpi.is_dissertation_cmpi_equation = True
weighted_cmpi.dimension_scorers_specified = False
weighted_cmpi.metric_status = "thesis_equation_dimension_scorers_unspecified"
weighted_cmpi.thesis_reference = "Ch.5 §5.1 p.131; §8.7.4 p.223 (CMPI equation); dimension scorers unspecified (C-EVAL-2)"
