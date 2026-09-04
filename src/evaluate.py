import argparse
import json
import math
import multiprocessing as mp
import warnings

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from tqdm import tqdm

from src.config import get_config, get_latest_weights
from src.data_pipeline import ChessMoveDataset, build_tokenized_arrays, get_or_build_tokenizer
from src.model import Transformer, get_model


ILLEGAL_MOVE_ERRORS = (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError, chess.InvalidMoveError)


def get_canonical_game_order(
        config: Dict,
        total_games: int
    ) -> np.ndarray:
    """Deterministic (seeded) game ordering shared by every evaluation metric.

    Every metric processes a PREFIX of this same fixed permutation, so
    "500 games" always means the same 500 games across metrics and
    across separate invocations -- and growing a metric from 500 to
    2,000 games later only requires evaluating games 500-2,000, not
    starting over. This is what makes the progress-file resume pattern
    below correct: the set of "already-done" games is always exactly
    the prefix a saved `num_games_done` describes.

    Args:
        config: A config dict (see `get_config`), used for `seed`.
        total_games (int): Number of games in the underlying dataset.

    Returns:
        np.ndarray: A permutation of `range(total_games)`.
    """
    rng = np.random.default_rng(config['seed'])
    return rng.permutation(total_games)


def load_progress(path: Path) -> Optional[Dict]:
    """Loads a metric's saved resume state, if any.

    Args:
        path (Path): Path to that metric's `progress.json`.

    Returns:
        Dict | None: The saved state, or None if no prior run exists.
    """
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_progress(
        path: Path,
        state: Dict
    ) -> None:
    """Saves a metric's resume state.

    Args:
        path (Path): Path to that metric's `progress.json`.
        state (Dict): The accumulator state to persist.
    """
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, 'w') as f:
        json.dump(state, f, indent = 2)


def classify_move_type(move_str: str) -> str:
    """Buckets a decoded SAN move string into one coarse category.

    Priority order (a move only ever gets one label): castling,
    promotion, capture, normal. A promotion-by-capture like "exd8=Q"
    is counted as a promotion, not a capture -- promotions are rarer
    and arguably more interesting to break out on their own.

    Args:
        move_str (str): A decoded move token, e.g. "Nf3", "O-O", "exd5".

    Returns:
        str: One of "castling", "promotion", "capture", "normal".
    """
    if move_str in ('O-O', 'O-O-O'):
        return 'castling'
    if '=' in move_str:
        return 'promotion'
    if 'x' in move_str:
        return 'capture'
    return 'normal'


def classify_phase(
        ply_index: int,
        phase_boundaries: Tuple[int, int]
    ) -> str:
    """Buckets a move's ply index into a coarse game phase.

    A fixed-ply approximation (not material-based) -- simple and cheap,
    good enough to reveal whether the model does much better in the
    small, memorizable space of common openings than in the wide-open
    middlegame.

    Args:
        ply_index (int): 0-indexed position of this move within the game.
        phase_boundaries (Tuple[int, int]): (opening_end, middlegame_end).

    Returns:
        str: One of "opening", "middlegame", "endgame".
    """
    opening_end, middlegame_end = phase_boundaries
    if ply_index < opening_end:
        return 'opening'
    if ply_index < middlegame_end:
        return 'middlegame'
    return 'endgame'


def _empty_core_state() -> Dict:
    return {
        'num_games_done': 0,
        'num_positions': 0,
        'num_correct': 0,
        'num_correct_top3': 0,
        'num_correct_top5': 0,
        'num_legal': 0,
        'probability_sum': 0.0,
        'neg_log_likelihood_sum': 0.0,
        'phase_counts': {'opening': [0, 0], 'middlegame': [0, 0], 'endgame': [0, 0]},
        'movetype_counts': {'capture': [0, 0], 'castling': [0, 0], 'promotion': [0, 0], 'normal': [0, 0]},
    }


def _finalize_core_state(state: Dict) -> Dict:
    n = state['num_positions']
    rate = lambda x: (x / n) if n else None
    return {
        'num_games': state['num_games_done'],
        'num_positions': n,
        'next_move_accuracy': rate(state['num_correct']),
        'top3_accuracy': rate(state['num_correct_top3']),
        'top5_accuracy': rate(state['num_correct_top5']),
        'legal_move_rate': rate(state['num_legal']),
        'mean_actual_move_probability': rate(state['probability_sum']),
        'perplexity': math.exp(state['neg_log_likelihood_sum'] / n) if n else None,
        'accuracy_by_phase': {k: (v[0] / v[1] if v[1] else None) for k, v in state['phase_counts'].items()},
        'accuracy_by_move_type': {k: (v[0] / v[1] if v[1] else None) for k, v in state['movetype_counts'].items()},
    }


def run_core_metrics(
        config: Dict,
        checkpoint_path: str,
        num_games: int,
        model: Optional[Transformer] = None,
        tokenizer: Optional[Tokenizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
    """Computes the cheap-to-evaluate metrics, resumably.

    "Cheap" = one batched forward pass per batch of games, no external
    engine calls: next-move accuracy, top-3/top-5 accuracy, legal-move
    rate (via python-chess replay of the real game), mean probability
    assigned to the actual move, perplexity, and next-move accuracy
    broken down by game phase and by move type. The [EOS]-target
    position is excluded from every metric, same reasoning as before:
    these are next-*move* metrics, not next-token-including-end-of-game
    metrics.

    Resumable like a training checkpoint: `num_games` is compared
    against a saved `progress.json` (a prefix length into the fixed
    order from `get_canonical_game_order`); only the new games beyond
    what's already been done are evaluated, and their contribution is
    merged into the saved running totals. Calling this again with a
    smaller `num_games` than already done is a no-op that just reports
    the existing (larger) totals -- evaluation never shrinks.

    Args:
        config: A config dict (see `get_config`).
        checkpoint_path (str): Path to the checkpoint being evaluated
            (used only to namespace where results are saved).
        num_games (int): Target number of games this call should cover.
        model (Transformer | None): Pass an already-loaded, eval()-mode
            model to skip reloading it (used by `run_evaluation` to
            share one load across metrics). Loaded from `checkpoint_path`
            if omitted.
        tokenizer (Tokenizer | None): Reused if given, else rebuilt.
        device (torch.device | None): Reused if given, else picked.

    Returns:
        Dict: The finalized metrics (see `_finalize_core_state`).
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tokenizer or get_or_build_tokenizer(config)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id('[PAD]')

    test_ids, test_lengths = build_tokenized_arrays(config, 'test', tokenizer)
    dataset = ChessMoveDataset(test_ids, test_lengths, tokenizer, config['context_size'])
    total_games = len(dataset)
    order = get_canonical_game_order(config, total_games)

    state = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)
    global_step = state['global_step']
    del state

    results_dir = Path(config['eval_results_dir']) / f'step_{global_step}' / 'core'
    progress_path = results_dir / 'progress.json'
    accumulator = load_progress(progress_path) or _empty_core_state()

    target = min(num_games, total_games)
    if target <= accumulator['num_games_done']:
        print(f"[core] already covers {accumulator['num_games_done']} games (>= requested {target}); nothing new to do.")
        return _finalize_core_state(accumulator)

    new_indices = order[accumulator['num_games_done']:target].tolist()
    dataloader = DataLoader(Subset(dataset, new_indices), batch_size = config['batch_size'], shuffle = False)

    if model is None:
        model = get_model(config, vocab_size).to(device)
        checkpoint_state = torch.load(checkpoint_path, map_location = device, weights_only = False)
        model.load_state_dict(checkpoint_state['model_state_dict'])
        model.eval()

    phase_boundaries = config['eval_phase_boundaries']

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = f"Core metrics ({accumulator['num_games_done']}->{target} games)", unit = ' batches'):
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            log_probs = model.project(model.decode(decoder_input, decoder_mask)) # (batch, context_size, vocab_size)
            top5 = log_probs.topk(5, dim = -1).indices # (batch, context_size, 5)

            label = label.cpu()
            top5 = top5.cpu()
            log_probs = log_probs.cpu()

            for row in range(label.shape[0]):
                real_tokens = label[row]
                top5_row = top5[row]
                row_log_probs = log_probs[row]

                num_real_and_eos = int((real_tokens != pad_id).sum().item())
                num_real_moves = num_real_and_eos - 1
                if num_real_moves <= 0:
                    continue

                board = chess.Board()
                for i in range(num_real_moves):
                    actual_id = int(real_tokens[i])
                    top5_ids = top5_row[i].tolist()
                    predicted_id = top5_ids[0]

                    accumulator['num_positions'] += 1
                    is_correct = int(predicted_id == actual_id)
                    accumulator['num_correct'] += is_correct
                    if actual_id in top5_ids[:3]:
                        accumulator['num_correct_top3'] += 1
                    if actual_id in top5_ids:
                        accumulator['num_correct_top5'] += 1
                    accumulator['probability_sum'] += row_log_probs[i, actual_id].exp().item()
                    accumulator['neg_log_likelihood_sum'] += -row_log_probs[i, actual_id].item()

                    predicted_move_str = tokenizer.id_to_token(predicted_id)
                    try:
                        board.parse_san(predicted_move_str)
                        accumulator['num_legal'] += 1
                    except ILLEGAL_MOVE_ERRORS:
                        pass

                    actual_move_str = tokenizer.id_to_token(actual_id)
                    phase = classify_phase(i, phase_boundaries)
                    move_type = classify_move_type(actual_move_str)
                    accumulator['phase_counts'][phase][0] += is_correct
                    accumulator['phase_counts'][phase][1] += 1
                    accumulator['movetype_counts'][move_type][0] += is_correct
                    accumulator['movetype_counts'][move_type][1] += 1

                    try:
                        board.push_san(actual_move_str)
                    except ILLEGAL_MOVE_ERRORS:
                        break

    accumulator['num_games_done'] = target
    save_progress(progress_path, accumulator)

    final = _finalize_core_state(accumulator)
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump({**final, 'checkpoint': str(checkpoint_path)}, f, indent = 2)
    return final


def _empty_centipawn_state() -> Dict:
    return {
        'num_games_done': 0,
        'num_positions': 0,
        'model_loss_sum': 0.0,
        'actual_loss_sum': 0.0,
        'num_model_illegal': 0,
    }


def _finalize_centipawn_state(state: Dict) -> Dict:
    n = state['num_positions']
    rate = lambda x: (x / n) if n else None
    return {
        'num_games': state['num_games_done'],
        'num_positions': n,
        'mean_model_centipawn_loss': rate(state['model_loss_sum']),
        'mean_actual_move_centipawn_loss': rate(state['actual_loss_sum']),
        'model_illegal_move_rate_in_sample': rate(state['num_model_illegal']),
    }


def _build_move_sequences(
        model: Transformer,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        pad_id: int,
        device: torch.device
    ) -> List[List[Tuple[str, str]]]:
    """Runs the model once to get (actual, predicted) move strings per game.

    Deliberately returns plain strings, not tensors -- this is the
    handoff point between the GPU/CPU model-inference side (this
    function) and the pure-`python-chess` worker side
    (`_centipawn_worker`), which runs in separate processes and has no
    need for torch at all. Keeping model inference in the main process
    means workers never touch CUDA state or the model.

    Args:
        model (Transformer): Trained model, already in eval() mode.
        dataloader (DataLoader): Yields batches from a ChessMoveDataset.
        tokenizer (Tokenizer): For decoding token ids to move strings.
        pad_id (int): The tokenizer's [PAD] token id.
        device (torch.device): Device to run inference on.

    Returns:
        List[List[Tuple[str, str]]]: One entry per game with at least
            one real move; each is a list of (actual_move_str,
            predicted_move_str) pairs in play order.
    """
    sequences = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = 'Preparing move sequences', unit = ' batches'):
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            predicted = model.project(model.decode(decoder_input, decoder_mask)).argmax(dim = -1)

            label = label.cpu()
            predicted = predicted.cpu()

            for row in range(label.shape[0]):
                real_tokens = label[row]
                predicted_tokens = predicted[row]

                num_real_and_eos = int((real_tokens != pad_id).sum().item())
                num_real_moves = num_real_and_eos - 1
                if num_real_moves <= 0:
                    continue

                sequences.append([
                    (tokenizer.id_to_token(int(real_tokens[i])), tokenizer.id_to_token(int(predicted_tokens[i])))
                    for i in range(num_real_moves)
                ])
    return sequences


def _chunk_list(
        items: List,
        num_chunks: int
    ) -> List[List]:
    """Splits `items` into at most `num_chunks` roughly-equal, contiguous pieces."""
    if not items:
        return []
    num_chunks = max(1, min(num_chunks, len(items)))
    size = math.ceil(len(items) / num_chunks)
    return [items[i:i + size] for i in range(0, len(items), size)]


def _centipawn_worker(
        args: Tuple[List[List[Tuple[str, str]]], str, int, int]
    ) -> Dict:
    """Worker process: opens one Stockfish engine and scores a chunk of games.

    Runs in a separate process (via `multiprocessing`), so it opens its
    own engine subprocess rather than sharing one — engines are cheap
    to start relative to how long a chunk of games takes to analyse.
    Pure `python-chess` + a UCI engine; no torch/model involved, which
    is why `_build_move_sequences` hands off plain move strings instead
    of tensors.

    Args:
        args: (move_sequences, stockfish_path, depth, max_loss) --
            bundled into one tuple because `multiprocessing.Pool.imap*`
            passes a single argument to the worker function.

    Returns:
        Dict: Partial accumulator with this chunk's contribution
            (`num_positions`, `model_loss_sum`, `actual_loss_sum`,
            `num_model_illegal`) -- merged into the running total by
            the caller.
    """
    move_sequences, stockfish_path, depth, max_loss = args
    limit = chess.engine.Limit(depth = depth)
    partial = {'num_positions': 0, 'model_loss_sum': 0.0, 'actual_loss_sum': 0.0, 'num_model_illegal': 0}

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        for moves in move_sequences:
            board = chess.Board()
            for actual_move_str, predicted_move_str in moves:
                mover_color = board.turn

                reference_info = engine.analyse(board, limit)
                reference_score = reference_info['score'].pov(mover_color).score(mate_score = 100_000)

                board_actual = board.copy()
                try:
                    board_actual.push_san(actual_move_str)
                except ILLEGAL_MOVE_ERRORS:
                    # Stray [UNK] real move (~0% on held-out data) -- stop this
                    # game's replay, same as the core metrics and legal-rate loop.
                    break
                actual_info = engine.analyse(board_actual, limit)
                actual_score = actual_info['score'].pov(mover_color).score(mate_score = 100_000)
                actual_loss = max(0, min(max_loss, reference_score - actual_score))

                board_model = board.copy()
                try:
                    board_model.push_san(predicted_move_str)
                    model_info = engine.analyse(board_model, limit)
                    model_score = model_info['score'].pov(mover_color).score(mate_score = 100_000)
                    model_loss = max(0, min(max_loss, reference_score - model_score))
                except ILLEGAL_MOVE_ERRORS:
                    model_loss = max_loss
                    partial['num_model_illegal'] += 1

                partial['num_positions'] += 1
                partial['model_loss_sum'] += model_loss
                partial['actual_loss_sum'] += actual_loss

                board.push_san(actual_move_str)
    finally:
        engine.quit()

    return partial


def run_centipawn_metrics(
        config: Dict,
        checkpoint_path: str,
        num_games: int,
        model: Optional[Transformer] = None,
        tokenizer: Optional[Tokenizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
    """Computes Stockfish-based centipawn loss, resumably.

    For each real-move position, runs Stockfish (at `config['stockfish_depth']`)
    on the position BEFORE the move to get an "almost ideal" reference
    evaluation (its own top choice), then separately evaluates the
    positions reached by (a) the move actually played and (b) the
    model's predicted move, each from the same mover's perspective.
    Centipawn loss = reference_eval - resulting_eval, clipped to
    [0, config['centipawn_max_loss']] (same convention as prior
    chess-LM evaluation work, e.g. Karvonen's Chess-GPT). An illegal
    model move is scored at exactly the clip value rather than excluded
    -- unplayable is at least as bad as the worst clipped blunder.

    Reports BOTH the model's mean loss and the actual (human) move's
    mean loss, deliberately: since the model is trained to imitate
    whatever move was actually played (not engine-optimal play), the
    meaningful comparison is model-vs-human, not model-vs-zero. A
    model tracking the human baseline is doing exactly what it was
    trained to do, even though neither number will be near zero.

    Real cost: up to 3 engine searches per move position (worse than
    `run_core_metrics`'s single forward pass), so this is opt-in via
    `config['centipawn_num_games']` and needs a separately installed
    Stockfish binary at `config['stockfish_path']`. Parallelized across
    `config['centipawn_num_workers']` processes, each running its own
    engine over a slice of the new games -- these per-position analyses
    are independent of each other, so this scales close to linearly
    with worker count, unlike deepening one search's own threading.

    Resumable exactly like `run_core_metrics` -- see that docstring.

    Args:
        config: A config dict (see `get_config`).
        checkpoint_path (str): Path to the checkpoint being evaluated.
        num_games (int): Target number of games this call should cover.
        model (Transformer | None): Reused if given, else loaded.
        tokenizer (Tokenizer | None): Reused if given, else rebuilt.
        device (torch.device | None): Reused if given, else picked.

    Returns:
        Dict: The finalized metrics (see `_finalize_centipawn_state`).
    """
    stockfish_path = config.get('stockfish_path')
    if not stockfish_path or not Path(stockfish_path).exists():
        raise FileNotFoundError(
            f"Stockfish binary not found at config['stockfish_path'] = {stockfish_path!r}. "
            "Install it (e.g. `winget install Stockfish.Stockfish` on Windows) and set this config key."
        )

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tokenizer or get_or_build_tokenizer(config)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id('[PAD]')

    test_ids, test_lengths = build_tokenized_arrays(config, 'test', tokenizer)
    dataset = ChessMoveDataset(test_ids, test_lengths, tokenizer, config['context_size'])
    total_games = len(dataset)
    order = get_canonical_game_order(config, total_games)

    state = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)
    global_step = state['global_step']
    del state

    results_dir = Path(config['eval_results_dir']) / f'step_{global_step}' / 'centipawn'
    progress_path = results_dir / 'progress.json'
    accumulator = load_progress(progress_path) or _empty_centipawn_state()

    target = min(num_games, total_games)
    if target <= accumulator['num_games_done']:
        print(f"[centipawn] already covers {accumulator['num_games_done']} games (>= requested {target}); nothing new to do.")
        return _finalize_centipawn_state(accumulator)

    new_indices = order[accumulator['num_games_done']:target].tolist()

    if model is None:
        model = get_model(config, vocab_size).to(device)
        checkpoint_state = torch.load(checkpoint_path, map_location = device, weights_only = False)
        model.load_state_dict(checkpoint_state['model_state_dict'])
        model.eval()

    dataloader = DataLoader(Subset(dataset, new_indices), batch_size = config['batch_size'], shuffle = False)
    move_sequences = _build_move_sequences(model, dataloader, tokenizer, pad_id, device)

    depth = config['stockfish_depth']
    max_loss = config['centipawn_max_loss']
    num_workers = max(1, config.get('centipawn_num_workers') or 1)

    # More chunks than workers so tqdm reflects real progress rather than
    # ticking once per worker; still few enough that engine-startup
    # overhead (one per chunk) stays negligible next to analysis time.
    chunks = _chunk_list(move_sequences, max(1, num_workers * 4))
    worker_args = [(chunk, stockfish_path, depth, max_loss) for chunk in chunks]

    desc = f"Centipawn loss ({accumulator['num_games_done']}->{target} games, {num_workers} workers)"
    partials = []
    if num_workers > 1 and len(chunks) > 1:
        with mp.Pool(processes = min(num_workers, len(chunks))) as pool:
            for partial in tqdm(pool.imap_unordered(_centipawn_worker, worker_args), total = len(worker_args), desc = desc, unit = ' chunks'):
                partials.append(partial)
    else:
        for args in tqdm(worker_args, desc = desc, unit = ' chunks'):
            partials.append(_centipawn_worker(args))

    for partial in partials:
        accumulator['num_positions'] += partial['num_positions']
        accumulator['model_loss_sum'] += partial['model_loss_sum']
        accumulator['actual_loss_sum'] += partial['actual_loss_sum']
        accumulator['num_model_illegal'] += partial['num_model_illegal']

    accumulator['num_games_done'] = target
    save_progress(progress_path, accumulator)

    final = _finalize_centipawn_state(accumulator)
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump({**final, 'checkpoint': str(checkpoint_path), 'stockfish_depth': depth, 'num_workers': num_workers}, f, indent = 2)
    return final


def save_example_predictions(
        model: Transformer,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        device: torch.device,
        results_dir: Path,
        num_examples: int = 10
    ) -> None:
    """Saves a handful of real-vs-predicted move sequences for manual inspection.

    Teacher-forced, same as the qualitative check `train.py` prints during
    training — useful here as a human-readable companion to the aggregate
    metrics, not a metric itself.

    Args:
        model (Transformer): Trained model, already in eval() mode.
        dataloader (DataLoader): Yields batches from a ChessMoveDataset.
        tokenizer (Tokenizer): For decoding token ids back to move strings.
        device (torch.device): Device to run inference on.
        results_dir (Path): Folder to write `example_predictions.json` into.
        num_examples (int): How many games (from one batch) to save.
    """
    pad_id = tokenizer.token_to_id('[PAD]')
    batch = next(iter(dataloader))

    with torch.no_grad():
        decoder_input = batch['decoder_input'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)

        predicted = model.project(model.decode(decoder_input, decoder_mask)).argmax(dim = -1)

    examples = []
    for row in range(min(num_examples, label.shape[0])):
        real_tokens = label[row].cpu()
        predicted_tokens = predicted[row].cpu()
        length = int((real_tokens != pad_id).sum().item())

        examples.append({
            'real_moves': [tokenizer.id_to_token(t) for t in real_tokens[:length].tolist()],
            'predicted_moves': [tokenizer.id_to_token(t) for t in predicted_tokens[:length].tolist()],
        })

    results_dir.mkdir(parents = True, exist_ok = True)
    with open(results_dir / 'example_predictions.json', 'w') as f:
        json.dump(examples, f, indent = 2)


def run_evaluation(
        config: Dict,
        checkpoint_path: Optional[str] = None,
        centipawn_num_games: Optional[int] = None
    ) -> Dict:
    """Loads a checkpoint once and runs every configured evaluation metric against it.

    Args:
        config: A config dict (see `get_config`).
        checkpoint_path (str | None): Path to a specific checkpoint to
            evaluate. Defaults to the latest checkpoint in
            `config['model_folder']`.
        centipawn_num_games (int | None): Overrides
            `config['centipawn_num_games']` for this call. 0 (the
            config default) skips the centipawn metric entirely.

    Returns:
        Dict: {'core': <core metrics>, 'centipawn': <centipawn metrics>}
            (the 'centipawn' key is only present if that metric ran).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}.")

    tokenizer = get_or_build_tokenizer(config)
    vocab_size = tokenizer.get_vocab_size()

    checkpoint_path = checkpoint_path or get_latest_weights(config)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in '{config['model_folder']}' to evaluate.")

    model = get_model(config, vocab_size).to(device)
    state = torch.load(checkpoint_path, map_location = device, weights_only = False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    global_step = state['global_step']
    print(f"Evaluating checkpoint '{checkpoint_path}' (trained to step {global_step}).")

    core = run_core_metrics(config, checkpoint_path, config['eval_num_games'], model = model, tokenizer = tokenizer, device = device)
    print(f"[core] games={core['num_games']} positions={core['num_positions']}")
    print(f"[core] next-move accuracy: {core['next_move_accuracy']:.4f} (top-3: {core['top3_accuracy']:.4f}, top-5: {core['top5_accuracy']:.4f})")
    print(f"[core] legal-move rate: {core['legal_move_rate']:.4f}")
    print(f"[core] mean probability on actual move: {core['mean_actual_move_probability']:.4f}")
    print(f"[core] perplexity: {core['perplexity']:.4f}")
    print(f"[core] accuracy by phase: {core['accuracy_by_phase']}")
    print(f"[core] accuracy by move type: {core['accuracy_by_move_type']}")

    results = {'core': core}

    centipawn_num_games = config['centipawn_num_games'] if centipawn_num_games is None else centipawn_num_games
    if centipawn_num_games:
        centipawn = run_centipawn_metrics(config, checkpoint_path, centipawn_num_games, model = model, tokenizer = tokenizer, device = device)
        print(f"[centipawn] games={centipawn['num_games']} positions={centipawn['num_positions']}")
        print(f"[centipawn] mean model-move loss: {centipawn['mean_model_centipawn_loss']:.1f} cp")
        print(f"[centipawn] mean actual (human) move loss: {centipawn['mean_actual_move_centipawn_loss']:.1f} cp")
        results['centipawn'] = centipawn

    test_ids, test_lengths = build_tokenized_arrays(config, 'test', tokenizer)
    dataset = ChessMoveDataset(test_ids, test_lengths, tokenizer, config['context_size'])
    order = get_canonical_game_order(config, len(dataset))
    example_loader = DataLoader(Subset(dataset, order[:10].tolist()), batch_size = 10, shuffle = False)
    save_example_predictions(model, example_loader, tokenizer, device, Path(config['eval_results_dir']) / f'step_{global_step}')

    return results


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description = 'Evaluate a chess move predictor checkpoint.')
    parser.add_argument('--checkpoint', type = str, default = None, help = 'Path to a specific checkpoint (defaults to the latest).')
    parser.add_argument('--num-games', type = int, default = None, help = "Override config['eval_num_games'] for the core metrics (use 0 for the full test split).")
    parser.add_argument('--centipawn-games', type = int, default = None, help = "Number of games for the Stockfish centipawn-loss metric (0 or omitted skips it).")
    parser.add_argument('--stockfish-path', type = str, default = None, help = "Path to the Stockfish binary (required only if --centipawn-games is set).")
    args = parser.parse_args()

    config = get_config()
    if args.num_games is not None:
        config['eval_num_games'] = None if args.num_games == 0 else args.num_games
    if args.stockfish_path is not None:
        config['stockfish_path'] = args.stockfish_path

    run_evaluation(config, checkpoint_path = args.checkpoint, centipawn_num_games = args.centipawn_games)
