# Legacy credentials and thesis migration

## Credential exposure

Previous public Git revisions contained plaintext database credentials and third-party API tokens in `Main.py`, `preprocess.py`, `Verb-net.py`, and `wiki.py`. This commit removes these values from the current source tree, but **does not erase them from Git history** and **does not revoke them with providers**.

**Immediate required actions by credential owners:**
1. Rotate the MySQL account password; ideally create a restricted `ansg_user` limited to a local research database. Review whether MySQL was ever network-accessible.
2. Revoke and reissue the Repustate API key using provider account support/console; watch billing/usage.
3. Revoke and reissue the Arabic NLP service token using its provider account/support (if still valid).
4. Inspect git history for exposed secrets and plan historical removal using GitHub's sensitive data removal procedure after rotations. Coordinate history rewriting because all clones/forks may retain old copies.
5. Store new credentials only in your local `.env`, secret manager, or CI secrets (never commit them). Python currently reads `os.environ` and does **not** automatically load `.env` files; export variables in the shell or use a local loader.

## Thesis alignment

The legacy scripts are **preprocessing/linguistic extraction precursors**, not CSAM/DAL/RL implementations. They support Chapter 3 and Arabic normalization/morphological features in Chapter 4. Their SQL tables and third-party NLP dependencies are not identical to ANHC schema, and no equivalence is asserted. New `src/ansg/` pipeline should consume explicitly validated corpus adapters, preserving annotations (DIA / MOTIF / MORAL), with no assumption old NLP annotations were the original training dataset.

## Security caveat

Changing files at branch head is not equivalent to revoking provider credentials. Do not claim that credentials are invalidated until provider-side confirmation.
