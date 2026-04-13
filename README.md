# Commit Regimes — Phase-Aware LR Controller

**Minimal reproducible phase-aware LR controller for CIFAR-10.**

From [Commit Regimes: Phase-Aware Fine-Tuning for Personality-Stable LoRA](https://thirdrail.world) — Third Rail Research.

## What This Is

This repository demonstrates a simple phase-aware training controller that monitors the smoothed loss derivative (δL) and applies phase-conditioned learning-rate interventions. On CIFAR-10, this controller can outperform a matched baseline in this setup under the same model, data split, and seed configuration.

The goal is to make the mechanism easy to inspect, reproduce, and extend.

## Quick Start

```bash
pip install torch torchvision matplotlib numpy
python phase_controller.py                  # single seed
python phase_controller.py --multi-seed     # 3 seeds + aggregated summary
```

Runtime: ~5-8 min per seed (GPU), ~15-20 min (CPU).

## The Mechanism

The controller monitors smoothed loss derivative (δL) each epoch and detects four phases:

| Phase | Trigger | Intervention |
|-------|---------|-------------|
| **S0_EXPLORE** | Training start | Standard LR (base schedule) |
| **S1_BOUNDARY** | δL drops below threshold | **LR pulse** (1.10x) — push through the transition |
| **S2_AXIS_LOCK** | δL stabilizes near zero | **LR decay** (0.5x), **batch scaling** (2x via grad accum) |
| **S3_POLISH** | Eval loss plateaus | **Aggressive LR decay** (0.1x) |

Both arms use an identical base LR schedule (warmup + cosine decay). The controller multiplies on top of it — all interventions are applied multiplicatively to the shared base schedule. The comparison is apples-to-apples.

The controller observes epoch N and recommends multipliers applied at epoch N+1. This one-epoch delay is intentional — the controller is reactive, not predictive.

## Current Scope

- **Implemented**: phase detection via δL, LR modulation (pulse at S1, decay at S2/S3), effective batch scaling via gradient accumulation
- **δL thresholds** (`--s1-threshold`, `--s2-threshold`) are tuned for this CIFAR-10 setup. Different models, loss scales, or augmentation strategies may require adjustment.
- **Intended as a minimal mechanism demo**, not a full benchmark suite

## Multi-Seed Validation

```bash
python phase_controller.py --multi-seed
python phase_controller.py --multi-seed --seeds "42,137,2024,7,314"
```

Runs both arms across all seeds, aggregates mean/std, writes `results/multi_seed_summary.json`.

### Success Criteria

A successful reproduction shows:
- Positive mean delta across seeds
- Majority of individual seeds showing improvement (positive delta)
- Visible S1→S2 transitions in the phase-aware arm's figure
- Phase-aware LR departing from baseline only after transition detection (not before)

## Reproducibility Notes

- All random sources seeded: `torch`, `numpy`, `random`, CUDA
- CUDA deterministic mode enabled when available (`cudnn.deterministic=True`, `cudnn.benchmark=False`)
- DataLoaders use seeded generators for identical batch order across arms
- Both arms share the exact same base LR schedule; controller multiplies on top
- Full per-epoch history (loss, accuracy, LR, phase, δL, grad accum) saved to JSON
- **Expected seed variance**: ±0.3-0.8% accuracy delta across seeds
- **Hardware note**: absolute numbers may shift between GPU/CPU or different GPU architectures; the relative delta between arms should usually preserve direction, though individual seeds and hardware differences can still affect magnitude

## CLI Options

```
--seed N              Random seed (default: 42)
--epochs N            Training epochs (default: 30)
--device {auto,cuda,mps,cpu}
--multi-seed          Run multiple seeds and aggregate
--seeds "42,137,2024" Comma-separated seeds for multi-seed mode
--s1-threshold F      δL threshold for S0→S1 (default: -0.02)
--s2-threshold F      |δL| threshold for S1→S2 (default: 0.005)
--lr-pulse F          LR multiplier during S1 (default: 1.10)
--output-dir PATH     Where to save figures and JSON (default: results/)
```

## From CIFAR to LoRA

The same controller architecture drives persona fine-tuning on Gemma 4 26B-A4B at production scale (detailed in the Commit Regimes paper). The key mechanisms transfer: δL phase detection, LR pulse at the susceptibility boundary, and TES-gated checkpointing. The CIFAR-10 demo isolates the mechanism; the paper shows it at scale.

## Outputs

Each run produces:
- `results/figure_seed_{N}.png` — 4-panel comparison chart with phase shading
- `results/results_seed_{N}.json` — full per-epoch history for both arms
- `results/multi_seed_summary.json` — aggregated mean/std across seeds (multi-seed mode)

## License

MIT — Third Rail / Convergent Labs

## Citation

```
Commit Regimes: Phase-Aware Fine-Tuning for Personality-Stable LoRA
Third Rail Research, 2026
https://thirdrail.world
```
