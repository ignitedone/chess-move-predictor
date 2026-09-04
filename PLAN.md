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

- [x] Add `chess` (python-chess) to `requirements.txt`
- [x] `src/evaluate.py`: load a checkpoint + tokenizer, run against a sample (or all) of the test split
- [x] Metric 1: next-move accuracy (`argmax(model_output)` vs. actual move, excluding the [EOS]-target position)
- [x] Metric 2: legal-move rate, via `board.parse_san()` against the real game's board state at each position
- [x] Metric 3: probability assigned to the move actually played (`exp(log_prob)`)
- [x] Save metrics + example predictions to disk, namespaced by checkpoint step (`eval_results/step_<N>/`) so evaluating multiple checkpoints never overwrites earlier results
- [x] Quick real checks (500 games) against epoch 1 and epoch 2 checkpoints: accuracy 40.0%->41.5%, legal-rate 94.9%->95.7%, mean probability 26.6%->27.9%
- [x] Redesigned to a resumable, per-metric "checkpoint" scheme: every metric processes a prefix of one fixed seeded game order and saves progress.json, so growing the sample size later only evaluates new games and merges rather than re-running
- [x] Added free-tier metrics: top-3/top-5 accuracy, perplexity, accuracy by game phase, accuracy by move type
- [x] Added Stockfish-based centipawn loss (model move vs. reference vs. actual human move), parallelized across worker processes; Stockfish 18 installed locally
- [x] **Decision: per-epoch evaluation scope.** Training continues through epoch 5 (see Stage 5). Rather than full-test-set core metrics on every epoch (~8-12h each under CPU contention with a concurrent centipawn run -- 5x that is more than the remaining time budget justifies), the scope is:
  - Epochs 1-4: core metrics on **half the test set** (229,102 of 458,204 games) + centipawn loss on 2,000 games each, giving a consistent, comparable progression table across epochs at a real but bounded cost.
  - Epoch 5 (final/best model): core metrics on the **full test set** (458,204 games) as the headline result, + centipawn loss on 2,000 games initially, with room to grow the centipawn sample further if time allows once everything else is done (the resumable per-metric design makes that a pure extension, not a redo).
  - Both metrics use the same fixed seeded game order (`get_canonical_game_order`) throughout, so every epoch's sample is a strict prefix of the next -- comparisons across epochs are apples-to-apples, and any sample can be grown later without re-evaluating games already covered.
  - [ ] Epoch 1 (step 128971): core metrics @ 229,102 games -- in progress (~92%+)
  - [x] Epoch 1 (step 128971): centipawn @ 2,000 games -- model 142.2cp / human 58.8cp loss, 5.18% illegal-in-sample
  - [ ] Epoch 2 (step 257942): core metrics @ 229,102 games -- in progress
  - [x] Epoch 2 (step 257942): centipawn @ 2,000 games -- model 129.9cp / human 58.6cp loss, 4.21% illegal-in-sample
  - [ ] Epoch 3 (step 386913): core metrics @ 229,102 games
  - [ ] Epoch 3 (step 386913): centipawn @ 2,000 games
  - [ ] Epoch 4: core metrics @ 229,102 games (once training completes)
  - [ ] Epoch 4: centipawn @ 2,000 games
  - [ ] Epoch 5 (final, if not skipped): core metrics @ 458,204 games (full test set)
  - [ ] Epoch 5 (final, if not skipped): centipawn @ 2,000 games, extend further if time allows
  - [x] **Decision: Stockfish search depth = 10 for every epoch's centipawn evaluation** (not 8) — the real epoch-1 run picked up `config.py`'s default of 10 before depth=8 was settled on for the dry runs; since depth 10 is a strictly better reference and the numbers were already computed, kept as the standard for consistency rather than redone at 8.

## Stage 5 — Kaggle

- [x] Kaggle account exists (30 GPU-hrs/week, 20 TPU-hrs/week)
- [x] New Notebook, set Accelerator to GPU T4 x2 (or P100) before running any cells
- [x] Push repo to GitHub (public — needed for unauthenticated `git clone` from Kaggle's batch/API-pushed kernel runs, since Kaggle Secrets aren't available outside the interactive UI)
- [x] First notebook cell: `!git clone` the repo
- [x] Attach the cached `.npy` arrays as a Kaggle Dataset (`chess-move-predictor-tokenized`)
- [x] Run training with checkpointing on via `kaggle kernels push` (batch execution); epoch 1 (128,971 steps) in 6h15m, loss 8.6 -> 2.45; epoch 2 (128,971 more steps) in 6h41m, loss -> 2.315 — see `REPORT_NOTES.md`
- [x] Checkpoints backed up 4-ways per epoch (local + local archive + Kaggle working dataset + Kaggle permanent-archive dataset); TensorBoard `runs/` data archived per epoch under `runs_archive/epoch<N>/` immediately after each run, before pushing the next version
- [x] Track GPU-hour usage against the ~30hrs/week budget — 10.99h used as of end of epoch 1, 19.01h remaining, quota refreshes 2026-09-05
- [x] **Decision: train through epoch 5, then stop.** Diminishing per-epoch loss gains are expected (epoch 1->2 already smaller than epoch 0->1 would have been) but ample GPU quota remains; epoch 5 is the planned final model for the report's headline evaluation.
  - [x] Epoch 3 — complete: 386,913/386,913 steps in 6h37m38s, loss -> 2.325 (val loss 2.104 -> 2.043, min 2.020); checkpoint + TensorBoard data secured
  - [ ] Epoch 4 — pushing now that epoch 3 is secured (target step 515,884); quota at push time: 5.60h remaining of this week's 30h, resets 2026-09-05T00:00:00 (~5h from push) -- expected to run past the reset, which is fine given checkpointing
  - [ ] Epoch 5 (final) — queued, launch once epoch 4 completes and its checkpoint/TensorBoard data are secured (may be skipped depending on how the evaluation backlog looks by then)

## Stage 6 — Report

- [ ] Pull "Opis problema" / "Skup podataka" / "Metode" forward from the proposal; expand "Metode" with implementation detail
- [ ] Loss/metric plots from TensorBoard
- [ ] Test-set metric values from Stage 4
- [ ] Discussion: model right/wrong patterns, `[UNK]`/rare-move limitation, game-ending-cause limitation
- [ ] Cite Karvonen's chess-GPT blog post
- [ ] Note the GitHub repo link in the report
- [ ] Fill in `README.md`: project summary, setup/reproduction steps, pointers to report/checkpoints/metrics
- [ ] (Optional) result-conditioning ablation, framed as an extension beyond the proposal's scope
