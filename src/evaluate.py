import argparse
import json
import warnings

from pathlib import Path
from typing import Dict, Optional

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from tqdm import tqdm

from src.config import get_config, get_latest_weights
from src.data_pipeline import ChessMoveDataset, build_tokenized_arrays, get_or_build_tokenizer
from src.model import Transformer, get_model


def get_eval_dataloader(
        config: Dict,
        tokenizer: Tokenizer
    ) -> DataLoader:
    """Builds the test-split DataLoader evaluate.py runs against.

    If `config['eval_num_games']` is set, a random (seeded, so the sample
    is reproducible run-to-run) subset of that many games is used instead
    of the full test split.

    Args:
        config: A config dict (see `get_config`).
        tokenizer (Tokenizer): The trained move-level tokenizer.

    Returns:
        DataLoader: yields batches shaped like `ChessMoveDataset` examples.
            Not shuffled — evaluation order doesn't matter and a fixed
            order makes any run easier to spot-check against another.
    """
    test_ids, test_lengths = build_tokenized_arrays(config, 'test', tokenizer)
    dataset = ChessMoveDataset(test_ids, test_lengths, tokenizer, config['context_size'])

    num_games = config.get('eval_num_games')
    if num_games is not None and num_games < len(dataset):
        rng = np.random.default_rng(config['seed'])
        indices = sorted(rng.choice(len(dataset), size = num_games, replace = False).tolist())
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size = config['batch_size'], shuffle = False)


def evaluate_model(
        model: Transformer,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        device: torch.device
    ) -> Dict:
    """Computes the three proposal metrics against a held-out set of games.

    Two passes per batch:
      1. One batched, teacher-forced forward pass on the GPU/CPU (whichever
         `device` is) — the same shape of computation as a training step,
         just without the backward pass — giving next-move accuracy and
         the probability assigned to the move actually played.
      2. A per-game, sequential CPU pass over those already-computed
         predictions that drives a real `chess.Board`, giving the
         legal-move rate. This can't be batched/vectorized: legality
         depends on the exact board state reached by the real game so
         far, which only a real board (pushed one real move at a time)
         knows.

    The position that predicts [EOS] (i.e. "the game ends here") is
    excluded from all three metrics — the proposal's metrics are about
    predicting the next *move*, not predicting when the game ends.

    Args:
        model (Transformer): Trained model, already in eval() mode.
        dataloader (DataLoader): Yields batches from a ChessMoveDataset
            (or a Subset of one).
        tokenizer (Tokenizer): For decoding token ids back to move
            strings ("Nf3", "O-O", "f8=Q", ...).
        device (torch.device): Device to run inference on.

    Returns:
        Dict: {
            'num_games': int, games with at least one real move,
            'num_positions': int, total real-move positions evaluated,
            'next_move_accuracy': float in [0, 1],
            'legal_move_rate': float in [0, 1],
            'mean_actual_move_probability': float in [0, 1],
        }
    """
    pad_id = tokenizer.token_to_id('[PAD]')

    num_games = 0
    num_positions = 0
    num_correct = 0
    num_legal = 0
    probability_sum = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc = 'Evaluating', unit = ' batches'):
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            log_probs = model.project(model.decode(decoder_input, decoder_mask)) # (batch, context_size, vocab_size)
            predicted = log_probs.argmax(dim = -1) # (batch, context_size)

            label = label.cpu()
            predicted = predicted.cpu()
            log_probs = log_probs.cpu()

            for row in range(label.shape[0]):
                real_tokens = label[row]
                predicted_tokens = predicted[row]
                row_log_probs = log_probs[row]

                # (real moves) + (1 for the trailing [EOS] target) are the
                # non-[PAD] positions; the moves themselves are everything
                # before that trailing [EOS].
                num_real_and_eos = int((real_tokens != pad_id).sum().item())
                num_real_moves = num_real_and_eos - 1
                if num_real_moves <= 0:
                    continue
                num_games += 1

                board = chess.Board()
                for i in range(num_real_moves):
                    actual_id = int(real_tokens[i])
                    predicted_id = int(predicted_tokens[i])

                    num_positions += 1
                    if predicted_id == actual_id:
                        num_correct += 1
                    probability_sum += row_log_probs[i, actual_id].exp().item()

                    predicted_move_str = tokenizer.id_to_token(predicted_id)
                    try:
                        board.parse_san(predicted_move_str)
                        num_legal += 1
                    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError, chess.InvalidMoveError):
                        pass

                    # Advance the board with the REAL move, so the next
                    # iteration's legality check sees the position the
                    # game was actually in — not a position the model's
                    # (possibly illegal) guess would have reached.
                    actual_move_str = tokenizer.id_to_token(actual_id)
                    try:
                        board.push_san(actual_move_str)
                    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError, chess.InvalidMoveError):
                        # Only expected for a stray [UNK] real move (~0% of
                        # tokens on held-out data) — stop this game's
                        # replay rather than evaluate positions built on
                        # a board state that no longer matches the game.
                        break

    return {
        'num_games': num_games,
        'num_positions': num_positions,
        'next_move_accuracy': num_correct / num_positions,
        'legal_move_rate': num_legal / num_positions,
        'mean_actual_move_probability': probability_sum / num_positions,
    }


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
        checkpoint_path: Optional[str] = None
    ) -> Dict:
    """Loads a checkpoint and evaluates it against the test split.

    Args:
        config: A config dict (see `get_config`).
        checkpoint_path (str | None): Path to a specific checkpoint to
            evaluate. Defaults to the latest checkpoint in
            `config['model_folder']`.

    Returns:
        Dict: The metrics dict returned by `evaluate_model`.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}.")

    tokenizer = get_or_build_tokenizer(config)
    vocab_size = tokenizer.get_vocab_size()

    dataloader = get_eval_dataloader(config, tokenizer)

    model = get_model(config, vocab_size).to(device)
    checkpoint_path = checkpoint_path or get_latest_weights(config)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in '{config['model_folder']}' to evaluate.")

    state = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    global_step = state['global_step']
    print(f"Evaluating checkpoint '{checkpoint_path}' (trained to step {global_step}).")

    metrics = evaluate_model(model, dataloader, tokenizer, device)

    print(f"Games evaluated: {metrics['num_games']} ({metrics['num_positions']} move positions)")
    print(f"Next-move accuracy: {metrics['next_move_accuracy']:.4f}")
    print(f"Legal-move rate: {metrics['legal_move_rate']:.4f}")
    print(f"Mean probability assigned to the actual move: {metrics['mean_actual_move_probability']:.4f}")

    # Namespaced by step, not overwritten by the next checkpoint's run --
    # comparing metrics across checkpoints (e.g. epoch 1 vs epoch 2) is the
    # point, and losing an earlier run's results the way an early
    # checkpointing bug once did is exactly what this avoids.
    results_dir = Path(config['eval_results_dir']) / f'step_{global_step}'
    results_dir.mkdir(parents = True, exist_ok = True)
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump({**metrics, 'checkpoint': str(checkpoint_path)}, f, indent = 2)

    save_example_predictions(model, dataloader, tokenizer, device, results_dir)

    return metrics


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description = 'Evaluate a chess move predictor checkpoint.')
    parser.add_argument('--checkpoint', type = str, default = None, help = 'Path to a specific checkpoint (defaults to the latest).')
    parser.add_argument('--num-games', type = int, default = None, help = "Override config['eval_num_games'] (use 0 for the full test split).")
    args = parser.parse_args()

    config = get_config()
    if args.num_games is not None:
        config['eval_num_games'] = None if args.num_games == 0 else args.num_games

    run_evaluation(config, checkpoint_path = args.checkpoint)
