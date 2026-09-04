from pathlib import Path
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """
    Returns:
        A static dictionary of project configuration variables:
            seed (int): random seed used everywhere for reproducibility,
                including the train/val/test split.
            raw_data_path (str): path to the downloaded raw CSV
                (WhiteElo, BlackElo, Result, transcript columns).
            processed_data_dir (str): folder holding the train/val/test
                split CSVs derived from raw_data_path.
            train_split (float): fraction of games assigned to training.
            val_split (float): fraction of games assigned to validation.
            test_split (float): fraction of games assigned to test.
            csv_chunksize (int): rows read per chunk when streaming CSVs,
                so the ~4GB raw file is never fully loaded into memory.
            tokenizer_file (str): path where the trained move-level
                tokenizer is saved/loaded.
            token_frequencies_file (str): path where each move token's
                training-split occurrence count is saved, as a byproduct
                of training the tokenizer (see `get_or_build_tokenizer`).
            min_frequency (int): minimum occurrences (in the training
                split) for a move token to get its own vocabulary entry;
                rarer moves are folded into [UNK].
            vocab_size (int): maximum vocabulary size for the tokenizer
                trainer. Chess SAN move tokens form a naturally bounded
                vocabulary, so this is set high enough to act as a no-op
                cap rather than actually truncating the vocabulary.
            context_size (int): fixed sequence length (in move tokens,
                including [SOS]/[EOS]) that every training example is
                padded/truncated to. Chosen from the training split's move-
                count distribution (p95 ~= 129 moves) as the point past
                which reducing truncation further costs disproportionately
                more padding/compute for quadratic-in-context attention
                cost; see REPORT_NOTES.md for the full percentile/tradeoff
                table this was picked from.
            model_dimension (int): embedding/model dimension (d_model).
            num_layers (int): number of decoder blocks stacked.
            num_heads (int): number of self-attention heads per block.
            feed_forward_dimension (int): hidden dimension of each block's
                feed-forward sublayer.
            dropout (float): dropout rate used throughout the model.

                model_dimension/num_layers/num_heads/feed_forward_dimension/
                dropout are provisional starting points (small, given the
                vocab is only 5,543 tokens) — see REPORT_NOTES.md for the
                compute-feasibility check they'll be finalized against
                before Stage 3 training begins in earnest.
            batch_size (int): training/validation batch size.
            learning_rate (float): Adam learning rate.
            max_steps (int): training stops once `global_step` reaches
                this. A ceiling, not a target — training is step-based
                (not epoch-based) specifically because one epoch over the
                8.25M-game training split is far larger than any single
                Kaggle session; the real stopping point is decided from
                actual Kaggle T4 throughput plus remaining time, not
                fixed in advance. See REPORT_NOTES.md.
            max_epochs (int | None): if set, overrides `max_steps` with
                `max_epochs * len(train_dataloader)` — the preferred way
                to target "train for N full passes over the data" instead
                of hand-computing a step count. Not set here (opt-in
                per-run override, same pattern as `max_train_seconds`).
            max_train_seconds (float | None): if set, training also stops
                once this many wall-clock seconds have elapsed since the
                start of this call to `train_model`, even if `max_steps`
                hasn't been reached — whichever limit is hit first wins.
                Not set here (defaults to None via `config.get`); pass it
                as a per-run override (e.g. on Kaggle, to bound a session
                by time before real steps/sec throughput is known).
            checkpoint_every_steps (int): how often (in steps) to save a
                checkpoint to `model_folder`. Kept short so even a brief
                calibration run saves at least one checkpoint.
            validate_every_steps (int): how often (in steps) to compute
                validation loss and print the qualitative next-move check.
            num_validation_examples (int): how many validation games to
                print predictions for at each validation interval.
            model_folder (str): folder where checkpoints are saved.
            model_basename (str): checkpoint filename prefix; full name is
                `{model_basename}{global_step}.pt`.
            preload (str | None): `'latest'` to resume from the most
                recent checkpoint in `model_folder`, a specific step
                (str) to resume from that checkpoint, or `None` to start
                fresh.
            experiment_name (str): TensorBoard log directory.
            eval_num_games (int | None): number of test-split games to
                sample for evaluate.py. None evaluates the full test
                split. Sampling is inference-only (no backward pass) and
                cheap relative to training, but the legal-move-rate
                metric drives a real python-chess board per game
                sequentially on CPU, so a full-test-split run is still
                far slower than a single forward pass would suggest.
            eval_results_dir (str): folder where evaluate.py saves its
                metrics JSON and example predictions.
    """
    return {
        "seed": 561,
        "raw_data_path": "data/raw/gt1_8kElo_all.csv",
        "processed_data_dir": "data/processed",
        "train_split": 0.90,
        "val_split": 0.05,
        "test_split": 0.05,
        "csv_chunksize": 100_000,
        "tokenizer_file": "data/tokenizer/tokenizer_chess.json",
        "token_frequencies_file": "data/tokenizer/tokenizer_chess.freq.json",
        "min_frequency": 1,
        "vocab_size": 1_000_000,
        "context_size": 128,
        "model_dimension": 256,
        "num_layers": 6,
        "num_heads": 8,
        "feed_forward_dimension": 1024,
        "dropout": 0.1,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "max_steps": 100_000,
        "checkpoint_every_steps": 200,
        "validate_every_steps": 200,
        "num_validation_examples": 3,
        "model_folder": "weights",
        "model_basename": "chess_transformer_",
        "preload": "latest",
        "experiment_name": "runs/chess_transformer",
        "eval_num_games": 10_000,
        "eval_results_dir": "eval_results",
    }


def get_weights_file_path(
        config,
        step: str
    ) -> str:
    """
    Get the path to a saved checkpoint for a given training step.

    Args:
        config: Config file.
        step (str): Training step (`global_step`) the checkpoint was saved at.

    Returns:
        str: Path to the checkpoint file.
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{step}.pt"

    return str(Path('.') / model_folder / model_filename)


def get_latest_weights(config) -> str:
    """
    Get the most recent saved checkpoint from `model_folder`, by step number.

    Args:
        config: Config file.

    Returns:
        str | None: Path to the latest checkpoint, or None if the folder
            has no matching checkpoints yet.
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filenames = list(Path(model_folder).glob(f"{model_basename}*"))

    if len(model_filenames) == 0:
        return None

    # Extracts the step int from the filename to sort by.
    def extract_step(filename):
        return int(filename.stem.split('_')[-1])

    model_filenames.sort(key = extract_step)

    return str(model_filenames[-1])
