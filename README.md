# Tic Tac Toe AI Arena

A compact Python project implementing a Tic Tac Toe environment and a variety of AI agents (random, exhaustive/minimax, Monte Carlo, Dynamic Programming, Q-Learning, SARSA, Expected SARSA, Double Q-Learning). It includes a pleasant Flet-based desktop UI to play vs AI, train RL agents with live visualizations, run tournaments, and compare agents.

## Highlights

- 9 built-in agents (Random, Exhaustive Search, Minimax α-β, Monte Carlo, Dynamic Programming, SARSA, Expected SARSA, Q-Learning, Double Q-Learning)
- Flet desktop UI: play vs AI, watch AI vs AI, train agents and visualize training curves and Q-value heatmaps
- Tournament and comparison tools (round-robin, head-to-head heatmaps, leaderboards)
- Training helpers that return histories for plotting (reward, steps, epsilon, Q-size)

## Repository structure

- `tictactoe_flet_app.py` — Flet UI wrapper and orchestration for play, training, tournament and comparisons (desktop app entry point).
- `tictactoe_game.py` — Core environment, agent implementations, tournament manager and optional Pygame GUI.
- `config/clean_settings.json` — (optional) configuration file the app will try to load for default training hyperparameters.

> Note: the repository may create a `__pycache__` folder when running.

## Requirements

- Python 3.8+ recommended
- The project uses these Python packages:
  - flet
  - numpy
  - matplotlib
  - seaborn
  - (optional) pygame — required only for the Pygame GUI in `tictactoe_game.py`

I've added a `requirements.txt` in the repo root with the minimal packages.

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the Flet desktop application:

```powershell
python tictactoe_flet_app.py
```

This will start the Flet app. Use the tabs to:

- "Play vs AI" — play as X against a selected AI opponent, or watch AI vs AI matches.
- "Training" — pick an RL agent, train it (background thread) and view reward/epsilon/Q-size plots and Q-value heatmaps.
- "Tournament" — select agents and run a round-robin tournament with visual summaries.
- "Compare" — train/evaluate selected agents and produce overlay curves and head-to-head heatmaps.

## Running command-line or headless scripts

`tictactoe_game.py` is runnable directly for quick benchmarks and examples. Examples:

```powershell
# Quick benchmark run (default DP training + tournament demo)
python tictactoe_game.py
```

You can also import classes in `tictactoe_game.py` from other scripts, for example:

```python
from tictactoe_game import TicTacToe, QLearningAgent, Tournament
# train/evaluate programmatically
```

## Configuration

`tictactoe_flet_app.py` attempts to read `config/clean_settings.json` to set default hyperparameters like `training_episodes`, `learning_rate`, `epsilon_decay` and `gamma`. Create that path with a JSON object to override defaults, e.g.:

```json
{
  "training_episodes": 3000,
  "learning_rate": 0.1,
  "epsilon_decay": 0.995,
  "gamma": 0.95
}
```

If the file is missing the app will fall back to sensible defaults.

## Developer notes

- Agent implementations are in `tictactoe_game.py`.
  - RL agents expose `train` and `train_with_history` (where available) and fill `q_table` or `q_table1/q_table2` for visuals.
  - Tournament uses round-robin via `Tournament.run_match` and builds head-to-head and leaderboard statistics.
- The Flet UI (`tictactoe_flet_app.py`) keeps an in-memory session registry `TRAINED_MODELS` so newly trained agents can be reused in the same app session.

Edge cases and tips:
- Some agent training (especially value-iteration / DP) may take noticeable CPU time — use lower `training_episodes` on slower machines.
- If you only need to evaluate algorithms quickly, prefer fewer episodes (e.g., 200–1000) to speed iteration.

## Optional: Pygame GUI

`tictactoe_game.py` contains a `TicTacToeGUI` class that needs `pygame`. Install `pygame` if you want to run it and then create a small script that instantiates `TicTacToeGUI(agent)` and calls `.run()`.

## Contributing

Contributions are welcome.
