import time
import warnings

from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import get_config, get_latest_weights, get_weights_file_path
from src.data_pipeline import ChessMoveDataset, build_tokenized_arrays, get_or_build_tokenizer
from src.model import Transformer, get_model


def get_dataloaders(
        config: Dict,
        tokenizer: Tokenizer
    ) -> Tuple[DataLoader, DataLoader]:
    """Builds the train and validation DataLoaders over the cached tokenized arrays.

    Args:
        config: A config dict (see `get_config`), providing `batch_size`
            and everything `build_tokenized_arrays` needs.
        tokenizer (Tokenizer): The trained move-level tokenizer.

    Returns:
        Tuple[DataLoader, DataLoader]: train and validation DataLoaders.
    """
    train_ids, train_lengths = build_tokenized_arrays(config, 'train', tokenizer)
    val_ids, val_lengths = build_tokenized_arrays(config, 'val', tokenizer)

    train_dataset = ChessMoveDataset(train_ids, train_lengths, tokenizer, config['context_size'])
    val_dataset = ChessMoveDataset(val_ids, val_lengths, tokenizer, config['context_size'])

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = True)

    return train_dataloader, val_dataloader


def run_validation(
        model: Transformer,
        val_dataloader: DataLoader,
        loss_function: nn.Module,
        vocab_size: int,
        device: torch.device,
        num_batches: int = 20
    ) -> float:
    """Computes mean validation loss over a handful of random validation batches.

    Only samples `num_batches` rather than the full validation split — this
    runs every `validate_every_steps` during training, so it needs to stay
    cheap relative to a training step.

    Args:
        model (Transformer): The model being trained.
        val_dataloader (DataLoader): Validation DataLoader.
        loss_function (nn.Module): Loss function (ignores [PAD] positions).
        vocab_size (int): Move-token vocabulary size.
        device (torch.device): Device to run on.
        num_batches (int): Number of validation batches to average over.

    Returns:
        float: Mean validation loss over the sampled batches.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= num_batches:
                break

            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            output = model.project(model.decode(decoder_input, decoder_mask)) # (batch, context_size, vocab_size)
            loss = loss_function(output.view(-1, vocab_size), label.view(-1))
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def run_qualitative_check(
        model: Transformer,
        val_dataloader: DataLoader,
        tokenizer: Tokenizer,
        device: torch.device,
        print_fn: Callable[[str], None],
        num_examples: int = 3
    ) -> None:
    """Prints real vs. predicted next moves for a few validation games.

    Teacher-forced (the model sees the real move history, not its own
    previous predictions) — cheap, single forward pass, and enough to
    visually sanity-check that predictions are becoming move-like as
    training progresses. Proper autoregressive generation and legality
    checking against a real board is Stage 4's job (`evaluate.py`), not
    this quick in-training check.

    Args:
        model (Transformer): The model being trained.
        val_dataloader (DataLoader): Validation DataLoader.
        tokenizer (Tokenizer): The trained move-level tokenizer, for
            decoding ids back to move strings.
        device (torch.device): Device to run on.
        print_fn (Callable[[str], None]): How to print each line (e.g.
            `tqdm.write`, so it doesn't clobber the progress bar).
        num_examples (int): How many games (from one batch) to print.
    """
    model.eval()
    pad_id = tokenizer.token_to_id('[PAD]')
    batch = next(iter(val_dataloader))

    with torch.no_grad():
        decoder_input = batch['decoder_input'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)

        output = model.project(model.decode(decoder_input, decoder_mask))
        predicted = output.argmax(dim = -1) # (batch, context_size)

    for i in range(min(num_examples, decoder_input.shape[0])):
        real_tokens = label[i]
        pred_tokens = predicted[i]
        length = int((real_tokens != pad_id).sum().item())

        # Only the last few real moves, not the whole (mostly padding) sequence.
        start = max(0, length - 5)
        real_moves = [tokenizer.id_to_token(t) for t in real_tokens[start:length].tolist()]
        pred_moves = [tokenizer.id_to_token(t) for t in pred_tokens[start:length].tolist()]

        print_fn(f"    real: {real_moves}")
        print_fn(f"    pred: {pred_moves}")

    model.train()


def train_model(config: Dict) -> None:
    """Trains the decoder-only chess move predictor.

    Step-based (not epoch-based): the training DataLoader is cycled
    indefinitely and progress is tracked/checkpointed/validated by
    `global_step`, since one epoch over the 8.25M-game training split
    vastly exceeds a single Kaggle session. Supports resuming from the
    latest checkpoint (`config['preload'] == 'latest'`), required for
    training across multiple capped Kaggle sessions.

    Args:
        config: A config dict (see `get_config`).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}.")

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    tokenizer = get_or_build_tokenizer(config)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id('[PAD]')

    train_dataloader, val_dataloader = get_dataloaders(config, tokenizer)

    model = get_model(config, vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])
    loss_function = nn.CrossEntropyLoss(ignore_index = pad_id).to(device)

    writer = SummaryWriter(config['experiment_name'])

    global_step = 0
    epoch = 0
    preload = config['preload']
    model_filename = get_latest_weights(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename, map_location = device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        epoch = state['epoch']
    else:
        print("No model to preload, starting from the beginning.")

    def save_checkpoint():
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, get_weights_file_path(config, global_step))

    model.train()
    train_iterator = iter(train_dataloader)
    progress = tqdm(total = config['max_steps'], initial = global_step, desc = 'Training', unit = ' steps')

    max_train_seconds = config.get('max_train_seconds')
    start_time = time.time()

    while global_step < config['max_steps']:
        if max_train_seconds is not None and time.time() - start_time > max_train_seconds:
            progress.write(f"Reached max_train_seconds ({max_train_seconds}s), stopping.")
            break

        try:
            batch = next(train_iterator)
        except StopIteration:
            # Training DataLoader exhausted (a full pass over the training
            # split) — start again from the top. Only relevant if a run
            # ever gets far enough to complete an epoch.
            epoch += 1
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        decoder_input = batch['decoder_input'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)

        output = model.project(model.decode(decoder_input, decoder_mask)) # (batch, context_size, vocab_size)
        loss = loss_function(output.view(-1, vocab_size), label.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        writer.add_scalar('train_loss', loss.item(), global_step)
        progress.set_postfix({'loss': f'{loss.item():.3f}', 'epoch': epoch})
        progress.update(1)

        if global_step % config['validate_every_steps'] == 0:
            val_loss = run_validation(model, val_dataloader, loss_function, vocab_size, device)
            writer.add_scalar('val_loss', val_loss, global_step)
            progress.write(f"  step {global_step}: val_loss={val_loss:.3f}")
            run_qualitative_check(model, val_dataloader, tokenizer, device, progress.write, config['num_validation_examples'])
            writer.flush()

        if global_step % config['checkpoint_every_steps'] == 0:
            save_checkpoint()

    progress.close()

    # Always save a final checkpoint, even if global_step didn't land
    # exactly on a checkpoint_every_steps boundary.
    save_checkpoint()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train_model(get_config())
