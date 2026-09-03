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

- [x] Copy reference building blocks into `src/model.py`: `InputEmbeddings`, `PositionalEncoding`, `LayerNormalization`, `MultiHeadAttentionBlock`, `FeedForwardBlock`, `ResidualConnection`, `ProjectionLayer`
- [x] Adapt `DecoderBlock`: keep causal self-attention + feed-forward sublayers, drop cross-attention and its residual connection
- [x] Add `model_dimension` (256), `num_layers` (6), `num_heads` (8), `feed_forward_dimension` (1024), `dropout` (0.1) to `config.py` as provisional starting values
- [x] Wire into a decoder-only `Transformer` class / `build_transformer()`/`get_model()` factory
- [x] Sanity check: real `ChessMoveDataset`/`DataLoader` batch through the model — output shape `(4, 128, 5543)` = `[batch, context_size, vocab_size]`, 7,582,631 total params
- [x] CPU throughput proxy measured (~400-460 tokens/sec, flat across batch size) — see `REPORT_NOTES.md`; confirms training must run on GPU but doesn't settle model size
- [x] **Decision: compute feasibility (real numbers).** Measured on Kaggle T4: 6.14 steps/s at batch_size=64/context_size=128 ≈ 50,300 tokens/sec ≈ 5.85 hours/epoch. Model size kept as-is (7.58M params); see `REPORT_NOTES.md`.

## Stage 3 — Training

- [x] `src/train.py`: training loop adapted from reference `train_model()`, causal-masked forward pass only
- [x] Loss: `nn.CrossEntropyLoss(ignore_index=pad_token_id)`, `pad_token_id` sourced from `config.py`
- [x] Adam optimizer, lr = 3e-4 (`config.py`)
- [x] Step-based (not epoch-based) checkpoint/validation/logging cadence, per Stage 2's feasibility check
- [x] TensorBoard: `train_loss` per step, validation loss per validation interval
- [x] Qualitative validation check: predict next move on a handful of held-out partial games at each validation interval, compare to actual
- [x] Checkpointing: model + optimizer state + epoch + global_step, saved to `weights/`; resume-from-latest verified (fresh run → resume-and-stop → resume-and-continue, all correct)
- [x] Local smoke test — ran 20-30 steps (not a full "epoch of 500 games": training is step-based now, so that framing doesn't apply) against the real train/val data on CPU; loss fell 8.68 → 6.61, confirming the loop, checkpointing, and resume logic all work before spending any Kaggle GPU time

## Stage 4 — Evaluation

- [ ] Add `python-chess` to `requirements.txt`
- [ ] `src/evaluate.py`: load a checkpoint + tokenizer, run against the test split
- [ ] Metric 1: next-move accuracy (`argmax(model_output)` vs. actual move)
- [ ] Metric 2: legal-move rate, via `chess.Board().legal_moves` replay
- [ ] Metric 3: probability assigned to the move actually played
- [ ] Save metrics + example predictions to disk

## Stage 5 — Kaggle

- [x] Kaggle account exists (30 GPU-hrs/week, 20 TPU-hrs/week)
- [x] New Notebook, set Accelerator to GPU T4 x2 (or P100) before running any cells
- [x] Push repo to GitHub (public — needed for unauthenticated `git clone` from Kaggle's batch/API-pushed kernel runs, since Kaggle Secrets aren't available outside the interactive UI)
- [x] First notebook cell: `!git clone` the repo
- [x] Attach the cached `.npy` arrays as a Kaggle Dataset (`chess-move-predictor-tokenized`)
- [x] Run training with checkpointing on via `kaggle kernels push` (batch execution); first full epoch (128,971 steps) completed in 6h15m, loss 8.6 -> 2.45 — see `REPORT_NOTES.md`
- [x] Track GPU-hour usage against the ~30hrs/week budget — 10.99h used as of end of epoch 1, 19.01h remaining, quota refreshes 2026-09-05

## Stage 6 — Report

- [ ] Pull "Opis problema" / "Skup podataka" / "Metode" forward from the proposal; expand "Metode" with implementation detail
- [ ] Loss/metric plots from TensorBoard
- [ ] Test-set metric values from Stage 4
- [ ] Discussion: model right/wrong patterns, `[UNK]`/rare-move limitation, game-ending-cause limitation
- [ ] Cite Karvonen's chess-GPT blog post
- [ ] Note the GitHub repo link in the report
- [ ] Fill in `README.md`: project summary, setup/reproduction steps, pointers to report/checkpoints/metrics
- [ ] (Optional) result-conditioning ablation, framed as an extension beyond the proposal's scope
