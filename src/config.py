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

        More fields (batch_size, learning_rate, etc.) will be added here
        as train.py is implemented.
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
    }
