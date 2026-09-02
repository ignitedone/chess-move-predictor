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

        More fields (batch_size, model_dimension, etc.) will be added here
        as model.py/train.py are implemented.
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
    }
