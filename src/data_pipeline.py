import re

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.trainers import WordLevelTrainer

# Matches a leading move-number prefix glued to a token, e.g. "1." or "23.".
MOVE_NUMBER_RE = re.compile(r'^\d+\.+')


def parse_transcript(transcript: str) -> List[str]:
    """Parses a PGN movetext transcript into an ordered list of SAN moves.

    The dataset's `transcript` column stores movetext like
    "1.e4 c5 2.Nf3 Nc6 3.Bb5 a6 ..." — move numbers are glued to the white
    move of each pair with no separating space, moves are whitespace
    separated, and there is no trailing game result token (e.g. "1-0").

    Args:
        transcript (str): Raw PGN movetext for a single game.

    Returns:
        List[str]: The moves in order, in SAN notation (e.g.
            ["e4", "c5", "Nf3", "Nc6", "Bb5", "a6", ...]), with move-number
            prefixes stripped.
    """
    moves = []

    # Move numbers are only glued to the front of a token, so stripping a
    # leading "<digits>." prefix from each whitespace-separated token is
    # enough to recover the bare SAN moves.
    for token in transcript.split():
        move = MOVE_NUMBER_RE.sub('', token)
        if move:
            moves.append(move)

    return moves


def build_splits(
        config: Dict[str, Any],
        force_rewrite: bool = False
    ) -> Tuple[Path, Path, Path]:
    """Splits the raw dataset CSV into train/val/test CSVs.

    Streams `config['raw_data_path']` in chunks (rather than loading the
    whole ~4GB file into memory) and routes each row to a split by drawing
    one uniform random number per row from a single seeded generator. Since
    the generator's state advances continuously across chunks, the same
    seed and chunk size always reproduce the same split, without needing a
    row count up front. With ~9M rows, the realized split sizes land
    essentially exactly on the configured ratios.

    A no-op if all three split files already exist and `force_rewrite` is
    False.

    Args:
        config: A config dict (see `get_config`), providing
            `raw_data_path`, `processed_data_dir`, `train_split`,
            `val_split`, `csv_chunksize`, and `seed`.
        force_rewrite (bool): If True, rebuild the split files even if
            they already exist.

    Returns:
        Tuple[Path, Path, Path]: Paths to the train, val and test CSVs
            under `processed_data_dir`.
    """
    processed_dir = Path(config['processed_data_dir'])
    split_paths = {
        'train': processed_dir / 'train.csv',
        'val': processed_dir / 'val.csv',
        'test': processed_dir / 'test.csv',
    }

    if not force_rewrite and all(path.exists() for path in split_paths.values()):
        return split_paths['train'], split_paths['val'], split_paths['test']

    processed_dir.mkdir(parents = True, exist_ok = True)

    rng = np.random.default_rng(config['seed'])
    train_threshold = config['train_split']
    val_threshold = train_threshold + config['val_split']

    # Track whether each split file has had its header written yet, so the
    # first chunk touching a split writes fresh (mode 'w') and later chunks
    # append (mode 'a').
    header_written = {name: False for name in split_paths}

    for chunk in pd.read_csv(config['raw_data_path'], chunksize = config['csv_chunksize']):
        draws = rng.random(len(chunk))
        assignment = np.where(draws < train_threshold, 'train', np.where(draws < val_threshold, 'val', 'test'))

        for name, path in split_paths.items():
            subset = chunk[assignment == name]
            if subset.empty:
                continue
            subset.to_csv(path, mode = 'a' if header_written[name] else 'w', header = not header_written[name], index = False)
            header_written[name] = True

    return split_paths['train'], split_paths['val'], split_paths['test']


def iter_move_sequences(
        csv_path: Path,
        chunksize: int
    ) -> Iterator[str]:
    """Streams a split CSV and yields each game's moves as one space-joined string.

    Only the `transcript` column is read from disk. Space-joining the
    parsed moves (rather than yielding the raw transcript) lets a
    whitespace/char-delimiter pre-tokenizer treat each move as one token,
    the same way the reference project treats each whitespace-separated
    word as one token.

    Args:
        csv_path (Path): Path to a split CSV (as produced by `build_splits`).
        chunksize (int): Rows read per chunk, to avoid loading the whole
            split into memory at once.

    Yields:
        str: A game's SAN moves, in order, separated by single spaces.
    """
    for chunk in pd.read_csv(csv_path, usecols = ['transcript'], chunksize = chunksize):
        for transcript in chunk['transcript']:
            yield ' '.join(parse_transcript(transcript))


def get_or_build_tokenizer(
        config: Dict[str, Any],
        force_rewrite: bool = False
    ) -> Tokenizer:
    """Gets the move-level tokenizer, building and saving it if needed.

    If `config['tokenizer_file']` doesn't exist yet, or `force_rewrite` is
    set, trains a word-level tokenizer over move tokens (e.g. "Nf3",
    "Qxe7+", "O-O") from the training split only, then saves it. Otherwise
    loads the existing tokenizer from file.

    Unknown tokens ([UNK]) appear at inference/eval time when a move never
    seen in training shows up. [PAD] fills sequences out to the model's
    context size. [SOS]/[EOS] mark the start/end of a game's move sequence.

    Args:
        config: A config dict (see `get_config`), providing
            `tokenizer_file`, `min_frequency`, `vocab_size`, and everything
            `build_splits` needs.
        force_rewrite (bool): If True, retrain and overwrite the tokenizer
            even if `tokenizer_file` already exists.

    Returns:
        Tokenizer: The move-level tokenizer, trained on (or loaded as
            already having been trained on) the training split's moves.
    """
    tokenizer_path = Path(config['tokenizer_file'])

    if not force_rewrite and tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        train_path, _, _ = build_splits(config)

        # Move tokens are already whitespace-separated by iter_move_sequences,
        # so a plain char-delimiter split on ' ' is enough to recover them —
        # no need for a linguistic pre-tokenizer.
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

        trainer = WordLevelTrainer(
            special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
            min_frequency = config['min_frequency'],
            vocab_size = config['vocab_size']
        )
        tokenizer.train_from_iterator(iter_move_sequences(train_path, config['csv_chunksize']), trainer = trainer)

        tokenizer_path.parent.mkdir(parents = True, exist_ok = True)
        tokenizer.save(str(tokenizer_path))

    print(f"Move vocabulary size: {tokenizer.get_vocab_size()}.")
    return tokenizer
