# Test fixtures — SYNTHETIC

Every file in this directory was written by hand for unit tests of the ANSG
reimplementation (2026). None of them is an excerpt of, or derived from:

* any edition of *One Thousand and One Nights* (only the two short frame
  formulae "بلغني أيها الملك السعيد" and "وأدرك شهرزاد الصباح فسكتت عن الكلام
  المباح" appear, to test formula detection);
* ArabicVerbNet (the XML only mirrors the element/attribute structure that the
  legacy `Verb-net.py` reads, plus standard VerbNet elements);
* ArabicNameNet (format unknown; the TSV only exercises the configurable adapter);
* the ANHC corpus, CKG or CST-Onto (identifiers like `motif:test_*` are invented
  test ids, not thesis motif identifiers);
* any gold annotation.

They must never be used as research data or reported as results.
