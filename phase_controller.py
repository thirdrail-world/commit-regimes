#!/usr/bin/env python3
"""
Phase-Aware LR Controller — Minimal Reproducible Example (CIFAR-10)
====================================================================

From "Commit Regimes: Phase-Aware Fine-Tuning for Personality-Stable LoRA"
Third Rail Research · https://thirdrail.world

Demonstrates phase-conditioned LR interventions driven by loss derivative
(δL) signals. On CIFAR-10, this controller can outperform a matched cosine
baseline under the same model, data split, and seed configuration.

The controller monitors smoothed δL to detect four phases:
  S0_EXPLORE  → high δL, model orienting
  S1_BOUNDARY → δL drops sharply → apply LR pulse (+10%)
  S2_AXIS_LOCK → δL stabilizes → reduce LR (0.5x), increase effective batch (2x via grad accum)
  S3_POLISH   → eval loss plateaus → aggressive LR decay (0.1x)

Usage:
  python phase_controller.py                        # single seed
  python phase_controller.py --seed 137             # different seed
  python phase_controller.py --multi-seed           # run seeds 42,137,2024 and aggregate
  python phase_controller.py --device cpu            # CPU-only (slower)

Requirements:
  pip install torch torchvision matplotlib numpy

Reproducibility notes:
  - All random sources seeded (torch, numpy, python random, CUDA)
  - CUDA deterministic mode enabled when available
  - DataLoaders use seeded generators for identical batch order across arms
  - Both arms share an identical base LR schedule; controller multiplies on top
  - Expected seed variance: ±0.3-0.8% accuracy delta across seeds
  - Hardware differences (GPU vs CPU, different GPUs) may shift absolute numbers
    but the relative delta between arms should hold direction

Current scope:
  - Implemented: phase detection + LR modulation + effective batch scaling (grad accum)
  - δL thresholds are tuned for this CIFAR-10 setup and may need adjustment
    for different models, loss scales, or augmentation strategies
  - Intended as a minimal mechanism demo, not a full benchmark suite

Expected runtime: ~5-8 minutes per seed on GPU, ~15-20 on CPU
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ── Reproducibility ───────────────────────────────────────────────────────

def seed_everything(seed: int):
    """Seed all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_seeded_generator(seed: int) -> torch.Generator:
    """Create a seeded generator for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class Config:
    """Frozen experiment config. Shared across both arms except controller."""
    seed: int = 42
    device: str = "auto"

    # Data
    batch_size: int = 128
    num_workers: int = 2

    # Model — narrow ResNet for fast experiments
    model_width: int = 32

    # Training — shared base schedule
    epochs: int = 30
    base_lr: float = 0.01
    weight_decay: float = 5e-4
    warmup_epochs: int = 2

    # Phase controller (experimental arm only)
    lr_pulse_multiplier: float = 1.10
    grad_accum_multiplier: int = 2     # effective batch scaling via gradient accumulation
    s1_deriv_threshold: float = -0.02  # tuned for this CIFAR-10 setup
    s2_deriv_threshold: float = 0.005  # tuned for this CIFAR-10 setup
    s3_patience: int = 5
    deriv_window: int = 3

    # Output
    output_dir: str = "results"


# ── Shared Base Schedule ──────────────────────────────────────────────────

def base_lr_schedule(epoch: int, config: Config) -> float:
    """
    Shared LR schedule used by BOTH arms.
    Warmup + cosine decay. The controller multiplies on top of this.
    """
    if epoch < config.warmup_epochs:
        return config.base_lr * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
    return config.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Phase Controller ──────────────────────────────────────────────────────

class PhaseController:
    """
    Four-phase training controller driven by smoothed loss derivative (δL).

    Observes epoch N → recommends multipliers applied at epoch N+1.
    This one-epoch delay is intentional: the controller is reactive, not predictive.

    Returns (lr_multiplier, grad_accum_steps) each epoch.
    """

    PHASES = ("S0_EXPLORE", "S1_BOUNDARY", "S2_AXIS_LOCK", "S3_POLISH")

    def __init__(self, config: Config):
        self.cfg = config
        self.phase = "S0_EXPLORE"
        self.phase_history: List[str] = []
        self.loss_history: List[float] = []
        self.deriv_history: List[float] = []
        self.transition_epochs: dict = {}
        self._best_eval_loss = float("inf")
        self._no_improve_count = 0

    def _smoothed_derivative(self) -> float:
        if len(self.loss_history) < 2:
            return 0.0
        window = min(self.cfg.deriv_window, len(self.loss_history) - 1)
        recent = self.loss_history[-window - 1:]
        derivs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        return sum(derivs) / len(derivs)

    def observe(self, train_loss: float, eval_loss: Optional[float] = None) -> Tuple[float, int]:
        """
        Observe this epoch's losses. Return (lr_multiplier, grad_accum_steps)
        to apply on the NEXT epoch.
        """
        self.loss_history.append(train_loss)
        deriv = self._smoothed_derivative()
        self.deriv_history.append(deriv)
        prev_phase = self.phase

        # Phase transitions
        if self.phase == "S0_EXPLORE":
            if len(self.loss_history) >= 3 and deriv < self.cfg.s1_deriv_threshold:
                self.phase = "S1_BOUNDARY"
                self.transition_epochs["S1"] = len(self.loss_history) - 1

        elif self.phase == "S1_BOUNDARY":
            if abs(deriv) < self.cfg.s2_deriv_threshold:
                self.phase = "S2_AXIS_LOCK"
                self.transition_epochs["S2"] = len(self.loss_history) - 1

        elif self.phase == "S2_AXIS_LOCK":
            if eval_loss is not None:
                if eval_loss < self._best_eval_loss - 1e-4:
                    self._best_eval_loss = eval_loss
                    self._no_improve_count = 0
                else:
                    self._no_improve_count += 1
                if self._no_improve_count >= self.cfg.s3_patience:
                    self.phase = "S3_POLISH"
                    self.transition_epochs["S3"] = len(self.loss_history) - 1

        # Compute multipliers for next epoch
        lr_mult = 1.0
        grad_accum = 1

        if self.phase == "S1_BOUNDARY":
            lr_mult = self.cfg.lr_pulse_multiplier
        elif self.phase == "S2_AXIS_LOCK":
            lr_mult = 0.5
            grad_accum = self.cfg.grad_accum_multiplier
        elif self.phase == "S3_POLISH":
            lr_mult = 0.1
            grad_accum = self.cfg.grad_accum_multiplier

        if prev_phase != self.phase:
            self.phase_history.append(
                f"Epoch {len(self.loss_history)-1}: {prev_phase} -> {self.phase} (dL={deriv:.4f})"
            )

        return lr_mult, grad_accum


# ── Model ─────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class SmallResNet(nn.Module):
    """Narrow ResNet-18 variant for fast CIFAR-10 experiments."""
    def __init__(self, w=32, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, w, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.layer1 = self._make_layer(w, w, 2, stride=1)
        self.layer2 = self._make_layer(w, w * 2, 2, stride=2)
        self.layer3 = self._make_layer(w * 2, w * 4, 2, stride=2)
        self.fc = nn.Linear(w * 4, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


# ── Data ──────────────────────────────────────────────────────────────────

def make_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train_tf, test_tf


def get_loaders(config: Config, seed: int):
    """Create fresh data loaders with a seeded generator for reproducible batch order."""
    train_tf, test_tf = make_transforms()
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    g = make_seeded_generator(seed)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        generator=g, worker_init_fn=lambda wid: np.random.seed(seed + wid),
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ── Training ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_accum_steps=1):
    """Train one epoch with optional gradient accumulation for effective batch scaling."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets) / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100.0 * correct / total


def run_conventional(config: Config, device):
    """Conventional training: shared base schedule, no phase awareness."""
    seed_everything(config.seed)
    train_loader, test_loader = get_loaders(config, config.seed)
    model = SmallResNet(w=config.model_width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.base_lr,
                          momentum=0.9, weight_decay=config.weight_decay)

    history = {"train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": [], "lr": []}
    best_acc = 0.0

    for epoch in range(config.epochs):
        lr = base_lr_schedule(epoch, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_loss, eval_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)
        history["lr"].append(lr)
        best_acc = max(best_acc, eval_acc)

        print(f"  [Conv] Epoch {epoch+1:2d}/{config.epochs}  "
              f"loss={train_loss:.4f}  acc={eval_acc:.2f}%  lr={lr:.6f}")

    history["best_acc"] = best_acc
    return history


def run_phase_aware(config: Config, device):
    """Phase-aware training: same base schedule, controller multiplies on top."""
    seed_everything(config.seed)
    train_loader, test_loader = get_loaders(config, config.seed)
    model = SmallResNet(w=config.model_width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.base_lr,
                          momentum=0.9, weight_decay=config.weight_decay)
    controller = PhaseController(config)

    history = {
        "train_loss": [], "train_acc": [], "eval_loss": [], "eval_acc": [],
        "lr": [], "phase": [], "deriv": [],
        "applied_grad_accum": [], "applied_lr_mult": [],
    }
    best_acc = 0.0
    # Controller recommendations (applied next epoch). Start neutral.
    next_lr_mult = 1.0
    next_grad_accum = 1

    for epoch in range(config.epochs):
        # Record what we're actually applying this epoch
        applied_lr_mult = next_lr_mult
        applied_grad_accum = next_grad_accum

        # Base schedule (identical to conventional arm)
        base_lr = base_lr_schedule(epoch, config)
        # Apply controller multiplier on top
        effective_lr = base_lr * applied_lr_mult
        for pg in optimizer.param_groups:
            pg["lr"] = effective_lr

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_accum_steps=applied_grad_accum,
        )
        eval_loss, eval_acc = evaluate(model, test_loader, criterion, device)

        # Controller observes this epoch, recommends for next epoch
        next_lr_mult, next_grad_accum = controller.observe(train_loss, eval_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)
        history["lr"].append(effective_lr)
        history["phase"].append(controller.phase)
        history["deriv"].append(controller.deriv_history[-1])
        history["applied_grad_accum"].append(applied_grad_accum)
        history["applied_lr_mult"].append(applied_lr_mult)
        best_acc = max(best_acc, eval_acc)

        print(f"  [Phase] Epoch {epoch+1:2d}/{config.epochs}  "
              f"loss={train_loss:.4f}  acc={eval_acc:.2f}%  "
              f"lr={effective_lr:.6f}  phase={controller.phase}  ga={applied_grad_accum}")

    history["best_acc"] = best_acc
    history["transitions"] = controller.phase_history
    history["transition_epochs"] = controller.transition_epochs
    return history, controller


# ── Visualization ─────────────────────────────────────────────────────────

def plot_results(conv, phase, controller, config, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Phase-Aware LR Controller vs Matched Baseline (CIFAR-10, seed={config.seed})",
        fontsize=13, fontweight="bold", y=0.98,
    )
    epochs = range(1, config.epochs + 1)
    c_conv, c_phase = "#2196F3", "#FF5722"
    phase_colors = {"S0_EXPLORE": "#E0E0E0", "S1_BOUNDARY": "#FFF9C4",
                    "S2_AXIS_LOCK": "#C8E6C9", "S3_POLISH": "#BBDEFB"}

    def shade(ax):
        for i, p in enumerate(phase["phase"]):
            ax.axvspan(i + 0.5, i + 1.5, alpha=0.3,
                       color=phase_colors.get(p, "#E0E0E0"), linewidth=0)

    # Test Accuracy
    ax = axes[0, 0]
    shade(ax)
    ax.plot(epochs, conv["eval_acc"], color=c_conv, lw=2,
            label=f"Baseline (best: {conv['best_acc']:.2f}%)")
    ax.plot(epochs, phase["eval_acc"], color=c_phase, lw=2,
            label=f"Phase-Aware (best: {phase['best_acc']:.2f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy"); ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Training Loss
    ax = axes[0, 1]
    shade(ax)
    ax.plot(epochs, conv["train_loss"], color=c_conv, lw=2, label="Baseline")
    ax.plot(epochs, phase["train_loss"], color=c_phase, lw=2, label="Phase-Aware")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Loss Derivative + Transitions
    ax = axes[1, 0]
    shade(ax)
    ax.plot(epochs, phase["deriv"], color=c_phase, lw=2, label="δL (smoothed)")
    ax.axhline(y=config.s1_deriv_threshold, color="red", ls="--", alpha=0.5,
               label=f"S1 threshold ({config.s1_deriv_threshold})")
    ax.axhline(y=0, color="gray", ls="-", alpha=0.3)
    for label, ep in controller.transition_epochs.items():
        ax.axvline(x=ep + 1, color="black", ls=":", alpha=0.7)
        ax.annotate(label, xy=(ep + 1, max(phase["deriv"]) * 0.8),
                    fontsize=9, fontweight="bold", ha="center")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss Derivative (δL)")
    ax.set_title("Phase Transitions via δL"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # LR Comparison
    ax = axes[1, 1]
    shade(ax)
    ax.plot(epochs, conv["lr"], color=c_conv, lw=2, label="Baseline (cosine)")
    ax.plot(epochs, phase["lr"], color=c_phase, lw=2, label="Phase-Aware (modulated)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule"); ax.legend(fontsize=9)
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(fc=c, alpha=0.5, label=l) for l, c in
                 [("S0: Explore", "#E0E0E0"), ("S1: Boundary", "#FFF9C4"),
                  ("S2: Axis Lock", "#C8E6C9"), ("S3: Polish", "#BBDEFB")]],
        loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {output_path}")


# ── Multi-Seed Aggregation ────────────────────────────────────────────────

def run_multi_seed(seeds: list, config: Config, device):
    """Run both arms across multiple seeds. Return aggregated results."""
    from dataclasses import replace
    all_results = []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"  SEED {seed}")
        print(f"{'='*50}")

        # Fresh config per seed — don't mutate the original
        seed_config = replace(config, seed=seed)

        print("\n  --- Conventional ---")
        conv = run_conventional(seed_config, device)

        print("\n  --- Phase-Aware ---")
        phase, ctrl = run_phase_aware(seed_config, device)

        delta = phase["best_acc"] - conv["best_acc"]
        result = {
            "seed": seed,
            "conventional_best": conv["best_acc"],
            "phase_aware_best": phase["best_acc"],
            "delta": round(delta, 2),
            "transitions": ctrl.phase_history,
        }
        all_results.append(result)
        print(f"\n  Seed {seed}: conv={conv['best_acc']:.2f}%  phase={phase['best_acc']:.2f}%  Δ={delta:+.2f}%")

        # Save per-seed figure
        out_dir = Path(seed_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_results(conv, phase, ctrl, seed_config, out_dir / f"figure_seed_{seed}.png")

    # Aggregate
    conv_accs = [r["conventional_best"] for r in all_results]
    phase_accs = [r["phase_aware_best"] for r in all_results]
    deltas = [r["delta"] for r in all_results]
    positive_count = sum(1 for d in deltas if d > 0)

    summary = {
        "seeds": seeds,
        "conventional": {
            "mean": round(np.mean(conv_accs), 2),
            "std": round(np.std(conv_accs), 2),
            "per_seed": conv_accs,
        },
        "phase_aware": {
            "mean": round(np.mean(phase_accs), 2),
            "std": round(np.std(phase_accs), 2),
            "per_seed": phase_accs,
        },
        "delta": {
            "mean": round(np.mean(deltas), 2),
            "median": round(float(np.median(deltas)), 2),
            "std": round(np.std(deltas), 2),
            "per_seed": deltas,
            "positive_count": positive_count,
            "total_seeds": len(seeds),
            "all_positive": all(d > 0 for d in deltas),
        },
        "per_seed_results": all_results,
    }

    print(f"\n{'='*60}")
    print(f"  MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  Conventional:  {summary['conventional']['mean']:.2f}% ± {summary['conventional']['std']:.2f}")
    print(f"  Phase-Aware:   {summary['phase_aware']['mean']:.2f}% ± {summary['phase_aware']['std']:.2f}")
    print(f"  Mean Delta:    {summary['delta']['mean']:+.2f}% ± {summary['delta']['std']:.2f}")
    print(f"  Median Delta:  {summary['delta']['median']:+.2f}%")
    print(f"  Positive:      {positive_count}/{len(seeds)}")
    print(f"{'='*60}")

    out_dir = Path(config.output_dir)
    with open(out_dir / "multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {out_dir / 'multi_seed_summary.json'}")

    return summary


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Commit Regimes — Phase-Aware LR Controller (CIFAR-10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run seeds 42, 137, 2024 and aggregate results")
    parser.add_argument("--seeds", type=str, default="42,137,2024",
                        help="Comma-separated seeds for --multi-seed")
    # Expose controller thresholds
    parser.add_argument("--s1-threshold", type=float, default=-0.02,
                        help="δL threshold for S0→S1 transition (default: -0.02, tuned for CIFAR-10)")
    parser.add_argument("--s2-threshold", type=float, default=0.005,
                        help="|δL| threshold for S1→S2 transition (default: 0.005)")
    parser.add_argument("--lr-pulse", type=float, default=1.10,
                        help="LR multiplier during S1 boundary phase (default: 1.10)")
    args = parser.parse_args()

    config = Config(
        seed=args.seed, epochs=args.epochs, output_dir=args.output_dir,
        s1_deriv_threshold=args.s1_threshold, s2_deriv_threshold=args.s2_threshold,
        lr_pulse_multiplier=args.lr_pulse,
    )

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"{'='*60}")
    print(f"  Commit Regimes — Phase-Aware LR Controller")
    print(f"  Third Rail Research · https://thirdrail.world")
    print(f"{'='*60}")
    print(f"  Device:         {device}")
    print(f"  Epochs:         {config.epochs}")
    print(f"  S1 threshold:   {config.s1_deriv_threshold}")
    print(f"  S2 threshold:   {config.s2_deriv_threshold}")
    print(f"  LR pulse:       {config.lr_pulse_multiplier}x")
    print(f"  Batch scaling:  {config.grad_accum_multiplier}x (grad accum)")
    print(f"  Deterministic:  True")
    print(f"{'='*60}\n")

    if args.multi_seed:
        seeds = [int(s) for s in args.seeds.split(",")]
        run_multi_seed(seeds, config, device)
    else:
        print(f"  Seed: {config.seed}\n")
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 40)
        print("  ARM 1: Conventional (matched baseline)")
        print("=" * 40)
        t0 = time.time()
        conv = run_conventional(config, device)
        conv_time = time.time() - t0

        print(f"\n{'='*40}")
        print(f"  ARM 2: Phase-Aware")
        print(f"{'='*40}")
        t0 = time.time()
        phase, ctrl = run_phase_aware(config, device)
        phase_time = time.time() - t0

        delta = phase["best_acc"] - conv["best_acc"]
        print(f"\n{'='*60}")
        print(f"  RESULTS (seed={config.seed})")
        print(f"{'='*60}")
        print(f"  Conventional:  {conv['best_acc']:.2f}%  ({conv_time:.1f}s)")
        print(f"  Phase-Aware:   {phase['best_acc']:.2f}%  ({phase_time:.1f}s)")
        print(f"  Delta:         {delta:+.2f}%")
        if ctrl.phase_history:
            print(f"  Transitions:")
            for t in ctrl.phase_history:
                print(f"    {t}")
        print(f"{'='*60}\n")

        plot_results(conv, phase, ctrl, config, out_dir / f"figure_seed_{config.seed}.png")

        # Full per-epoch history
        results = {
            "config": {
                "seed": config.seed, "epochs": config.epochs,
                "device": str(device), "base_lr": config.base_lr,
                "lr_pulse_multiplier": config.lr_pulse_multiplier,
                "grad_accum_multiplier": config.grad_accum_multiplier,
                "s1_deriv_threshold": config.s1_deriv_threshold,
                "s2_deriv_threshold": config.s2_deriv_threshold,
                "deterministic": True,
            },
            "conventional": {
                "best_acc": conv["best_acc"], "time_seconds": round(conv_time, 1),
                "per_epoch": {k: v for k, v in conv.items() if k != "best_acc"},
            },
            "phase_aware": {
                "best_acc": phase["best_acc"], "time_seconds": round(phase_time, 1),
                "transitions": ctrl.phase_history,
                "transition_epochs": ctrl.transition_epochs,
                "per_epoch": {k: v for k, v in phase.items()
                              if k not in ("best_acc", "transitions", "transition_epochs")},
            },
            "delta_best_acc": round(delta, 2),
        }
        with open(out_dir / f"results_seed_{config.seed}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results saved: {out_dir / f'results_seed_{config.seed}.json'}")


if __name__ == "__main__":
    main()
