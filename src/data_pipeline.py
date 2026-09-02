import json
import re

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

# Matches a leading move-number prefix glued to a token, e.g. "1." or "23.".
MOVE_NUMBER_RE = re.compile(r'^\d+\.+')

# Matches a trailing check ("+") or checkmate ("#") annotation.
CHECK_SUFFIX_RE = re.compile(r'[+#]$')


def parse_transcript(transcript: str) -> List[str]:
    """Parses a PGN movetext transcript into an ordered list of SAN moves.

    The dataset's `transcript` column stores movetext like
    "1.e4 c5 2.Nf3 Nc6 3.Bb5 a6 ..." — move numbers are glued to the white
    move of each pair with no separating space, moves are whitespace
    separated, and there is no trailing game result token (e.g. "1-0").

    Trailing check ("+")/checkmate ("#") annotations are stripped, so e.g.
    "Qh8#" and "Qh8" become the same move token. Without this, a move and
    its check/checkmate variant are unrelated tokenizer vocabulary entries
    with no shared embedding, which fragments the vocabulary (measured at
    ~62% of vocab being +/# variants) far more than it helps — see
    REPORT_NOTES.md for the full reasoning.

    Args:
        transcript (str): Raw PGN movetext for a single game.

    Returns:
        List[str]: The moves in order, in SAN notation (e.g.
            ["e4", "c5", "Nf3", "Nc6", "Bb5", "a6", ...]), with move-number
            prefixes and check/checkmate suffixes stripped.
    """
    moves = []

    # Move numbers are only glued to the front of a token, so stripping a
    # leading "<digits>." prefix from each whitespace-separated token is
    # enough to recover the bare SAN moves.
    for token in transcript.split():
        move = MOVE_NUMBER_RE.sub('', token)
        move = CHECK_SUFFIX_RE.sub('', move)
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

    progress = tqdm(desc = 'Splitting raw dataset', unit = ' rows', unit_scale = True)
    for chunk in pd.read_csv(config['raw_data_path'], chunksize = config['csv_chunksize']):
        draws = rng.random(len(chunk))
        assignment = np.where(draws < train_threshold, 'train', np.where(draws < val_threshold, 'val', 'test'))

        for name, path in split_paths.items():
            subset = chunk[assignment == name]
            if subset.empty:
                continue
            subset.to_csv(path, mode = 'a' if header_written[name] else 'w', header = not header_written[name], index = False)
            header_written[name] = True

        progress.update(len(chunk))
    progress.close()

    return split_paths['train'], split_paths['val'], split_paths['test']


def iter_move_sequences(
        csv_path: Path,
        chunksize: int,
        desc: str = None
    ) -> Iterator[str]:
    """Streams a split CSV and yields each game's moves as one space-joined string.

    Only the `transcript` column is read from disk. Space-joining the
    parsed moves (rather than yielding the raw transcript) lets a
    whitespace/char-delimiter pre-tokenizer treat each move as one token,
    the same way the reference project treats each whitespace-separated
    word as one token.

    Shows a live tqdm progress bar (games processed so far, rate) as the
    CSV is streamed, since a full split is tens of millions of rows and
    silently running for tens of minutes is indistinguishable from a hang
    without one. No `total` is set (the split's row count isn't known
    without a separate scan), so the bar shows count/rate rather than a
    percentage/ETA.

    Args:
        csv_path (Path): Path to a split CSV (as produced by `build_splits`).
        chunksize (int): Rows read per chunk, to avoid loading the whole
            split into memory at once.
        desc (str): Label shown on the progress bar. Defaults to the
            CSV's filename.

    Yields:
        str: A game's SAN moves, in order, separated by single spaces.
    """
    progress = tqdm(desc = desc or f'Reading {Path(csv_path).name}', unit = ' games', unit_scale = True)
    for chunk in pd.read_csv(csv_path, usecols = ['transcript'], chunksize = chunksize):
        for transcript in chunk['transcript']:
            yield ' '.join(parse_transcript(transcript))
        progress.update(len(chunk))
    progress.close()


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

    Whenever the tokenizer is actually (re)trained, this also tallies each
    move token's occurrence count across the training split in the same
    corpus pass the trainer consumes, and saves it to
    `config['token_frequencies_file']` — reasoning about vocabulary policy
    (e.g. `min_frequency`) needs these counts, and computing them as a
    byproduct of training avoids a second full scan of the training split.

    Args:
        config: A config dict (see `get_config`), providing
            `tokenizer_file`, `token_frequencies_file`, `min_frequency`,
            `vocab_size`, and everything `build_splits` needs.
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

        frequencies: Counter = Counter()

        def counted_sequences() -> Iterator[str]:
            for sequence in iter_move_sequences(train_path, config['csv_chunksize'], desc = 'Training tokenizer'):
                frequencies.update(sequence.split(' '))
                yield sequence

        tokenizer.train_from_iterator(counted_sequences(), trainer = trainer)

        tokenizer_path.parent.mkdir(parents = True, exist_ok = True)
        tokenizer.save(str(tokenizer_path))

        frequencies_path = Path(config['token_frequencies_file'])
        frequencies_path.parent.mkdir(parents = True, exist_ok = True)
        with open(frequencies_path, 'w') as f:
            json.dump(frequencies, f)

    print(f"Move vocabulary size: {tokenizer.get_vocab_size()}.")
    return tokenizer


def get_token_frequencies(config: Dict[str, Any]) -> Dict[str, int]:
    """Loads the training-split move-token occurrence counts saved by `get_or_build_tokenizer`.

    Args:
        config: A config dict (see `get_config`), providing
            `token_frequencies_file`.

    Returns:
        Dict[str, int]: Move token -> number of occurrences in the
            training split, for every token the tokenizer trainer saw
            (including ones filtered out of the final vocabulary by
            `min_frequency`/`vocab_size`).
    """
    with open(config['token_frequencies_file']) as f:
        return json.load(f)


def build_tokenized_arrays(
        config: Dict[str, Any],
        split: str,
        tokenizer: Tokenizer,
        force_rewrite: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenizes an entire split once and caches it as packed NumPy arrays.

    Holding one Python list of ints per game in memory doesn't scale to
    millions of games (~8.25M in the training split alone) — Python int/list
    object overhead would put the training split's ~623M move tokens at
    roughly 20+ GB of RAM. Instead, every game's move-token ids are
    concatenated into one flat array (`token_ids`), alongside a parallel
    `lengths` array recording how many tokens belong to each game — enough
    to slice out game i's tokens in O(1) via a running offset (see
    `ChessMoveDataset`), at a small fraction of the memory cost.

    [SOS]/[EOS]/[PAD] are NOT included in `token_ids` — `ChessMoveDataset`
    adds those per example at a given `context_size`, so these cached
    arrays stay reusable across `context_size` changes.

    A no-op if cached arrays already exist for `split` and `force_rewrite`
    is False.

    Args:
        config: A config dict (see `get_config`), providing
            `processed_data_dir` and everything `build_splits` needs.
        split (str): One of 'train', 'val', 'test'.
        tokenizer (Tokenizer): The trained move-level tokenizer (see
            `get_or_build_tokenizer`), used to tokenize each game and to
            pick a compact-enough dtype for `token_ids`.
        force_rewrite (bool): If True, retokenize and overwrite the cached
            arrays even if they already exist.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            token_ids: 1D array of every game's move-token ids,
                concatenated in split order.
            lengths: 1D array, one entry per game, giving how many
                tokens (into token_ids) belong to that game.
    """
    processed_dir = Path(config['processed_data_dir'])
    token_ids_path = processed_dir / f'{split}_token_ids.npy'
    lengths_path = processed_dir / f'{split}_lengths.npy'

    if not force_rewrite and token_ids_path.exists() and lengths_path.exists():
        return np.load(token_ids_path), np.load(lengths_path)

    train_path, val_path, test_path = build_splits(config)
    csv_path = {'train': train_path, 'val': val_path, 'test': test_path}[split]

    # uint16 comfortably covers chess's naturally small move vocabulary;
    # fall back to a wider dtype only if an unusually large vocab demands it.
    ids_dtype = np.uint16 if tokenizer.get_vocab_size() <= np.iinfo(np.uint16).max else np.int32

    ids_chunks = []
    lengths_list = []
    for sequence in iter_move_sequences(csv_path, config['csv_chunksize'], desc = f'Tokenizing {split}'):
        ids = tokenizer.encode(sequence).ids
        ids_chunks.append(np.array(ids, dtype = ids_dtype))
        lengths_list.append(len(ids))

    token_ids = np.concatenate(ids_chunks)
    lengths = np.array(lengths_list, dtype = np.int32)

    processed_dir.mkdir(parents = True, exist_ok = True)
    np.save(token_ids_path, token_ids)
    np.save(lengths_path, lengths)

    print(f"Tokenized '{split}' split: {len(lengths)} games, {len(token_ids)} move tokens.")
    return token_ids, lengths


def causal_mask(size: int) -> torch.Tensor:
    """Generates a causal mask for the decoder's self-attention.

    A triangular matrix of ones that blocks each position from attending
    to positions after it, so predictions can only depend on moves already
    played.

    Args:
        size (int): Size of the mask matrix (equal to context_size).

    Returns:
        torch.Tensor: Boolean tensor of dimension (1, size, size).
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0


class ChessMoveDataset(TorchDataset):
    """Decoder-only next-move-prediction dataset over tokenized chess games.

    Adapted from the reference project's BilingualDataset, stripped down to
    the decoder-only path: no encoder_input/encoder_mask and no
    source_text/target_text/source_tokenizer/target_tokenizer split — chess
    move sequences are a single "language", not a source/target pair.

    Each game's raw move-token ids (see `build_tokenized_arrays`) are framed
    as one training example:
        decoder_input = [SOS, move_1, ..., move_K, PAD, ..., PAD]
        label         = [move_1, ..., move_K, EOS, PAD, ..., PAD]
    both of length `context_size`, so position t of decoder_input always
    predicts position t of label — the model learns "given the game so
    far, what's the next move" at every position simultaneously. Games
    longer than context_size - 1 moves have their tail truncated (the
    opening/middlegame up to that point still trains normally); see
    REPORT_NOTES.md for why context_size was chosen where it was.
    """

    def __init__(
            self,
            token_ids: np.ndarray,
            lengths: np.ndarray,
            tokenizer: Tokenizer,
            context_size: int
        ) -> None:
        """Initializing the ChessMoveDataset object.

        Args:
            token_ids (np.ndarray): 1D array of every game's move-token
                ids, concatenated in order (see `build_tokenized_arrays`).
            lengths (np.ndarray): 1D array, one entry per game, giving how
                many tokens (into token_ids) belong to that game.
            tokenizer (Tokenizer): The trained move-level tokenizer, used
                only to look up the [SOS]/[EOS]/[PAD] special-token ids.
            context_size (int): Fixed sequence length every example is
                padded/truncated to.
        """
        super().__init__()

        self.token_ids = token_ids
        self.lengths = lengths

        # Exclusive prefix sum: offsets[i] is where game i's tokens start in token_ids.
        self.offsets = np.concatenate(([0], np.cumsum(lengths, dtype = np.int64)[:-1]))

        self.context_size = context_size

        self.sos_token = tokenizer.token_to_id('[SOS]')
        self.eos_token = tokenizer.token_to_id('[EOS]')
        self.pad_token = tokenizer.token_to_id('[PAD]')

    def __len__(self) -> int:
        """
        Returns:
            int: Number of games in the dataset.
        """
        return len(self.lengths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Builds one training example for the game at `index`.

        Args:
            index (int): Index of the game to return.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with 3 fields:
                decoder_input:
                    Input to be fed to the decoder.
                    Tensor of dimension (context_size).
                decoder_mask:
                    Mask for the decoder, that masks any padding tokens and
                    disallows attending to future positions.
                    Tensor of dimension (1, context_size, context_size).
                label:
                    Expected model output (decoder_input shifted by one
                    move).
                    Tensor of dimension (context_size).
        """
        start = int(self.offsets[index])

        # A game longer than context_size - 1 moves has its tail truncated;
        # everything up to the cutoff still trains normally.
        max_moves = self.context_size - 1
        length = min(int(self.lengths[index]), max_moves)

        move_tokens = torch.tensor(self.token_ids[start:start + length].astype(np.int64), dtype = torch.int64)
        num_padding_tokens = max_moves - length
        pad = torch.full((num_padding_tokens,), self.pad_token, dtype = torch.int64)

        # decoder_input is [SOS] move[1] ... move[K] [PAD] ... [PAD].
        decoder_input = torch.cat([torch.tensor([self.sos_token], dtype = torch.int64), move_tokens, pad])

        # label is move[1] ... move[K] [EOS] [PAD] ... [PAD].
        label = torch.cat([move_tokens, torch.tensor([self.eos_token], dtype = torch.int64), pad])

        assert decoder_input.size(0) == self.context_size
        assert label.size(0) == self.context_size

        return {
            "decoder_input": decoder_input,
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.context_size),
            "label": label,
        }
