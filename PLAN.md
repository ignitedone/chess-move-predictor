# chess_move_predictor — Project Plan

Work top to bottom. Log the reasoning behind any nontrivial decision in `REPORT_NOTES.md` when it's made.

## Stage 1 — Data pipeline

- [x] Download `gt1_8kElo_all.zip`, load into pandas, inspect shape/columns
- [x] `parse_transcript()` — PGN string → flat SAN move list
- [x] `build_splits()` — train/val/test split (90/5/5, seed 561), saved under `data/processed/`
- [x] `get_or_build_tokenizer()` — word-level tokenizer trained on the train split only, saved to `data/tokenizer/tokenizer_chess.json` (also now saves per-token training-split frequency counts to `data/tokenizer/tokenizer_chess.freq.json` as a byproduct, via `get_token_frequencies()`)
- [x] `context_size` chosen from move-length percentiles, documented in `REPORT_NOTES.md`, set to 128 in `config.py`
- [x] `build_tokenized_arrays()` and `ChessMoveDataset` implemented in `src/data_pipeline.py`
- [x] **Decision: tokenizer vocabulary policy** — strip `+`/`#` (check/checkmate annotations), keep `min_frequency=1`. Reasoning and measurements in `REPORT_NOTES.md`.
- [x] `parse_transcript()` strips `+`/`#` via `CHECK_SUFFIX_RE`
- [x] Rebuild the tokenizer with `force_rewrite=True`; new vocab size 5,543, `[UNK]` rate ~0% on val/test — see `REPORT_NOTES.md`
- [x] Run `build_tokenized_arrays()` for train/val/test — cached under `data/processed/` (train: 8,254,144 games, 623,023,053 tokens)
- [x] Instantiate `ChessMoveDataset` per split; decoded a real example back to move strings and confirmed `label[t] == decoder_input[t+1]` for every real-move position (the one designed exception: `decoder_input` never contains `[EOS]`, it just pads after the last move)

## Stage 2 — Model

- [ ] Copy reference building blocks into `src/model.py`: `InputEmbeddings`, `PositionalEncoding`, `LayerNormalization`, `MultiHeadAttentionBlock`, `FeedForwardBlock`, `ResidualConnection`, `ProjectionLayer`
- [ ] Adapt `DecoderBlock`: keep causal self-attention + feed-forward sublayers, drop cross-attention and its residual connection
- [ ] Add `d_model`, `num_layers`, `num_heads`, `d_ff`, `dropout` to `config.py`
- [ ] Wire into a decoder-only `Transformer` class / `build_transformer()` factory
- [ ] Sanity check: dummy batch of random token ids through the model, confirm output shape `[batch, context_size, vocab_size]`
- [ ] **Decision: compute feasibility.** Measure throughput (tokens/sec) for a candidate model size and batch size on a Kaggle T4; estimate steps-per-epoch over ~623M training tokens against the ~30 GPU-hr/week budget; use this to finalize `d_model`/`num_layers`/`num_heads` and to decide Stage 3's checkpoint/validation cadence. Record in `REPORT_NOTES.md`.

## Stage 3 — Training

- [ ] `src/train.py`: training loop adapted from reference `train_model()`, causal-masked forward pass only
- [ ] Loss: `nn.CrossEntropyLoss(ignore_index=pad_token_id)`, `pad_token_id` sourced from `config.py`
- [ ] Adam optimizer, lr ≈ 3e-4 (add to `config.py`)
- [ ] Step-based (not epoch-based) checkpoint/validation/logging cadence, per Stage 2's feasibility check
- [ ] TensorBoard: `train_loss` per step, validation loss per validation interval
- [ ] Qualitative validation check: predict next move on a handful of held-out partial games at each validation interval, compare to actual
- [ ] Checkpointing: model + optimizer state + epoch + global_step, saved to `weights/`; must support resuming from the latest checkpoint
- [ ] Local smoke test (~500 games, 1 epoch, CPU) before spending any Kaggle GPU time

## Stage 4 — Evaluation

- [ ] Add `python-chess` to `requirements.txt`
- [ ] `src/evaluate.py`: load a checkpoint + tokenizer, run against the test split
- [ ] Metric 1: next-move accuracy (`argmax(model_output)` vs. actual move)
- [ ] Metric 2: legal-move rate, via `chess.Board().legal_moves` replay
- [ ] Metric 3: probability assigned to the move actually played
- [ ] Save metrics + example predictions to disk

## Stage 5 — Kaggle

- [ ] Create Kaggle account, verify phone number
- [ ] New Notebook, set Accelerator to GPU T4 x2 (or P100) before running any cells
- [ ] Push repo to GitHub (private is fine)
- [ ] First notebook cell: `!git clone` the repo
- [ ] Re-run the data pipeline on Kaggle, or attach the cached `.npy` arrays as a Kaggle Dataset
- [ ] Run training with checkpointing on; "Save Version" after any run whose checkpoints matter
- [ ] Track GPU-hour usage against the ~30hrs/week budget

## Stage 6 — Report

- [ ] Pull "Opis problema" / "Skup podataka" / "Metode" forward from the proposal; expand "Metode" with implementation detail
- [ ] Loss/metric plots from TensorBoard
- [ ] Test-set metric values from Stage 4
- [ ] Discussion: model right/wrong patterns, `[UNK]`/rare-move limitation, game-ending-cause limitation
- [ ] Cite Karvonen's chess-GPT blog post
- [ ] Note the GitHub repo link in the report
- [ ] Fill in `README.md`: project summary, setup/reproduction steps, pointers to report/checkpoints/metrics
- [ ] (Optional) result-conditioning ablation, framed as an extension beyond the proposal's scope
