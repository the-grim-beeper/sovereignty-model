# Sovereignty Economic Model

A three-layer formal economic model of the argument in "Two Kinds of Sovereignty."

## Quick Start

```bash
cd sovereignty-model
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Run Dashboard

```bash
streamlit run app/dashboard.py --server.port 3001
```

## Generate Figures

```bash
python generate_figures.py
```

Outputs to `output/figures/`.

## Architecture

- **Layer 1 (`model/analytical.py`)** — Optimal control: state equations, forward simulation, shadow prices, optimal α*
- **Layer 2 (`model/evolutionary.py`)** — Agent-based: N heterogeneous firms, innovation lottery, selection, Monte Carlo sweep
- **Layer 3 (`model/policy.py`)** — Policy interface: regime comparison, delay costs, domain scoring, integrated recommendations
- **Shared (`model/calibration.py`)** — Parameters with empirical anchors, h(σ) hump shape, parameter sweep utilities

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| D₀ (initial dependency) | 0.67 | Synergy Research Group |
| R (EU R&D/GDP) | 2.2% | European Investment Bank |
| σ* (optimal constraint) | 2.0 | Derived from h(σ) peak |
