"""
Tic Tac Toe AI Arena - Flet UI
- 9 agents
- Play vs AI
- Training (post-run visualizations)
- Tournament (round-robin) with static visualizations
"""

import os
import io
import json
import time
import base64
import threading
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import flet as ft

# Reuse core game and agents from the existing module
from tictactoe_game import (
    TicTacToe,
    Player,
    GameResult,
    BaseAgent,
    RandomAgent,
    ExhaustiveSearchAgent,
    MinMaxAgent,
    MonteCarloAgent,
    DynamicProgrammingAgent,
    QLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
    DoubleQLearningAgent,
    Tournament,
)

# In-memory registry of trained models (per app session)
TRAINED_MODELS: Dict[str, BaseAgent] = {}

# ------------------------ Configuration ------------------------
def load_clean_config() -> Dict[str, float]:
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "clean_settings.json")
    default = {"training_episodes": 2000, "learning_rate": 0.1, "epsilon_decay": 0.995, "gamma": 0.95}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            default.update({k: data.get(k, default[k]) for k in default})
    except Exception:
        pass
    return default

CFG = load_clean_config()

# ------------------------ UI styling helpers ------------------------
def heading(text: str) -> ft.Text:
    """Section heading text with consistent style."""
    return ft.Text(text, size=16, weight=ft.FontWeight.W_600)

def subheading(text: str) -> ft.Text:
    return ft.Text(text, size=14, weight=ft.FontWeight.W_500)

def stats_text(text: str) -> ft.Text:
    # Slightly larger, readable text used for KPI/stat blocks
    return ft.Text(text, size=13, selectable=True)

def card(title: Optional[str], controls: List[ft.Control]) -> ft.Card:
    content: List[ft.Control] = []
    if title:
        content.append(heading(title))
    content.extend(controls)
    return ft.Card(
        elevation=1,
        content=ft.Container(
            ft.Column(content, spacing=12),
            padding=16,
        ),
    )

def clone_trained_agent(name: str, player: Player) -> Optional[BaseAgent]:
    """Return a fresh agent instance with learned tables copied from registry if available."""
    ag = TRAINED_MODELS.get(name)
    if ag is None:
        return None
    clone = create_agent_by_name(name, player)
    # Copy hyperparameters when present
    for attr in ("alpha", "gamma", "epsilon"):
        if hasattr(ag, attr) and hasattr(clone, attr):
            setattr(clone, attr, getattr(ag, attr))
    # Copy Q structures
    if hasattr(ag, "q_table") and hasattr(clone, "q_table"):
        clone.q_table = dict(getattr(ag, "q_table"))
    if hasattr(ag, "q_table1") and hasattr(clone, "q_table1"):
        clone.q_table1 = dict(getattr(ag, "q_table1"))
    if hasattr(ag, "q_table2") and hasattr(clone, "q_table2"):
        clone.q_table2 = dict(getattr(ag, "q_table2"))
    return clone

def clone_agent_instance(src: BaseAgent, player: Player) -> BaseAgent:
    """Clone a given agent instance to a new player side, copying learned tables when present."""
    clone = create_agent_by_name(src.name, player)
    for attr in ("alpha", "gamma", "epsilon"):
        if hasattr(src, attr) and hasattr(clone, attr):
            setattr(clone, attr, getattr(src, attr))
    # Copy Q structures when relevant
    if hasattr(src, "q_table") and hasattr(clone, "q_table"):
        clone.q_table = dict(getattr(src, "q_table"))
    if hasattr(src, "q_table1") and hasattr(clone, "q_table1"):
        clone.q_table1 = dict(getattr(src, "q_table1"))
    if hasattr(src, "q_table2") and hasattr(clone, "q_table2"):
        clone.q_table2 = dict(getattr(src, "q_table2"))
    return clone

# ------------------------ Helpers ------------------------

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def create_agent_by_name(name: str, player: Player) -> BaseAgent:
    if name == "Random":
        return RandomAgent(player)
    if name == "Exhaustive Search":
        return ExhaustiveSearchAgent(player)
    if name == "Minimax α-β":
        return MinMaxAgent(player)
    if name == "Monte Carlo":
        # Use config hyperparameters when available
        ag = MonteCarloAgent(player, epsilon=1.0, alpha=float(CFG.get("learning_rate", 0.1)), gamma=float(CFG.get("gamma", 0.95)))
        return ag
    if name == "Dynamic Programming":
        ag = DynamicProgrammingAgent(player, gamma=float(CFG.get("gamma", 0.95)))
        return ag
    if name == "SARSA":
        ag = SARSAAgent(player, epsilon=0.1, alpha=float(CFG.get("learning_rate", 0.1)), gamma=float(CFG.get("gamma", 0.95)))
        return ag
    if name == "Expected SARSA" or name == "Expected-SARSA":
        ag = ExpectedSARSAAgent(player, epsilon=0.1, alpha=float(CFG.get("learning_rate", 0.1)), gamma=float(CFG.get("gamma", 0.95)))
        return ag
    if name == "Q-Learning":
        ag = QLearningAgent(player, epsilon=0.1, alpha=float(CFG.get("learning_rate", 0.1)), gamma=float(CFG.get("gamma", 0.95)))
        return ag
    if name == "Double Q-Learning":
        ag = DoubleQLearningAgent(player, epsilon=0.1, alpha=float(CFG.get("learning_rate", 0.1)), gamma=float(CFG.get("gamma", 0.95)))
        return ag
    return RandomAgent(player)


TRAINABLE = ["Q-Learning", "SARSA", "Expected SARSA", "Double Q-Learning", "Monte Carlo", "Dynamic Programming"]
ALL_AGENTS = [
    "Random",
    "Exhaustive Search",
    "Minimax α-β",
    "Monte Carlo",
    "Dynamic Programming",
    "SARSA",
    "Expected SARSA",
    "Q-Learning",
    "Double Q-Learning",
]

# ------------------------ Training logic ------------------------

def train_agent_collect_history(agent: BaseAgent, episodes: int = 2000, epsilon_start: float = 1.0,
                                epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                                opp_agent: Optional[BaseAgent] = None,
                                stop_event: Optional[threading.Event] = None):
    """Train the given RL agent and collect metrics.
    Returns a dict with rewards, steps, epsilons, q_table_size.
    """
    # Only RL agents are expected here
    history = {"rewards": [], "steps": [], "epsilons": [], "q_sizes": []}

    # Configure epsilon when supported
    if hasattr(agent, "epsilon"):
        agent.epsilon = epsilon_start

    # Choose opponent: if none provided, default to Random
    opp = opp_agent if opp_agent is not None else RandomAgent(Player.O if agent.player == Player.X else Player.X)
    # Ensure opponent has the opposite side
    if hasattr(opp, 'player'):
        opp.player = Player.O if agent.player == Player.X else Player.X

    for ep in range(episodes):
        if stop_event is not None and stop_event.is_set():
            break
        game = TicTacToe()
        # If agent plays O, opponent goes first once
        if agent.player == Player.O:
            a0 = opp.choose_action(game)
            game.make_move(a0)
        ep_steps = 0
        ep_reward = 0.0

        if isinstance(agent, QLearningAgent) or isinstance(agent, ExpectedSARSAAgent) or isinstance(agent, DoubleQLearningAgent):
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                state = game.get_state_key()
                avail = game.get_available_actions()
                if not avail:
                    break
                act = agent.choose_action(game)
                _, _, done, res = game.make_move(act)
                ep_steps += 1

                if done:
                    if res == GameResult.X_WIN and agent.player == Player.X:
                        ep_reward = 1.0
                    elif res == GameResult.O_WIN and agent.player == Player.O:
                        ep_reward = 1.0
                    elif res == GameResult.DRAW:
                        ep_reward = 0.5
                    else:
                        ep_reward = 0.0
                    # terminal update
                    if isinstance(agent, ExpectedSARSAAgent):
                        agent.update_q(state, act, ep_reward, "", [], True)
                    elif isinstance(agent, DoubleQLearningAgent):
                        agent.update_q(state, act, ep_reward, "", [], True)
                    else:
                        agent.update_q(state, act, ep_reward, game.get_state_key(), [], True)
                    break

                if stop_event is not None and stop_event.is_set():
                    break
                opp_act = opp.choose_action(game)
                _, _, opp_done, opp_res = game.make_move(opp_act)
                next_state = game.get_state_key()
                next_actions = game.get_available_actions()

                if opp_done:
                    r = -1.0 if ((opp_res == GameResult.X_WIN and agent.player == Player.O) or (opp_res == GameResult.O_WIN and agent.player == Player.X)) else 0.5
                    ep_reward = 0.0 if r < 0 else 0.5
                    agent.update_q(state, act, r, next_state, next_actions, True)
                    break
                else:
                    agent.update_q(state, act, 0.0, next_state, next_actions, False)

        elif isinstance(agent, SARSAAgent):
            state = game.get_state_key()
            act = agent.choose_action(game)
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                _, _, done, res = game.make_move(act)
                ep_steps += 1
                if done:
                    if res == GameResult.X_WIN and agent.player == Player.X:
                        ep_reward = 1.0
                    elif res == GameResult.O_WIN and agent.player == Player.O:
                        ep_reward = 1.0
                    elif res == GameResult.DRAW:
                        ep_reward = 0.5
                    else:
                        ep_reward = 0.0
                    agent.update_q(state, act, ep_reward, "", None, True)
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                opp_act = opp.choose_action(game)
                _, _, opp_done, opp_res = game.make_move(opp_act)
                next_state = game.get_state_key()
                if opp_done:
                    r = -1.0 if ((opp_res == GameResult.X_WIN and agent.player == Player.O) or (opp_res == GameResult.O_WIN and agent.player == Player.X)) else 0.5
                    ep_reward = 0.0 if r < 0 else 0.5
                    agent.update_q(state, act, r, next_state, None, True)
                    break
                next_act = agent.choose_action(game)
                agent.update_q(state, act, 0.0, next_state, next_act, False)
                state, act = next_state, next_act

        # epsilon decay
        if hasattr(agent, "epsilon"):
            agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)
            history["epsilons"].append(agent.epsilon)
        else:
            history["epsilons"].append(0.0)

        # q table size
        if isinstance(agent, QLearningAgent) or isinstance(agent, SARSAAgent) or isinstance(agent, ExpectedSARSAAgent):
            qsize = len(agent.q_table)
        elif isinstance(agent, DoubleQLearningAgent):
            qsize = len(agent.q_table1) + len(agent.q_table2)
        else:
            qsize = 0
        history["q_sizes"].append(qsize)

        history["rewards"].append(ep_reward)
        history["steps"].append(ep_steps)

    return history


def eval_agent_collect_history(agent: BaseAgent, episodes: int = 2000,
                               opp_agent: Optional[BaseAgent] = None) -> Dict[str, List[float]]:
    """Collect non-learning evaluation curves for an agent.
    Plays episodes versus an opponent (default Random) without updating any tables.
    Returns history dict compatible with plot/summary: rewards, steps, epsilons, q_sizes.
    """
    history = {"rewards": [], "steps": [], "epsilons": [], "q_sizes": []}
    # Ensure agent plays X for consistency
    agent.player = Player.X
    opp = opp_agent if opp_agent is not None else RandomAgent(Player.O)
    if hasattr(opp, 'player'):
        opp.player = Player.O
    game = TicTacToe()
    for _ in range(episodes):
        game.reset()
        ep_steps = 0
        # Play a full game without learning
        while game.check_winner() == GameResult.IN_PROGRESS:
            if game.current_player == Player.X:
                action = agent.choose_action(game)
            else:
                action = opp.choose_action(game)
            game.make_move(action)
            ep_steps += 1
        res = game.check_winner()
        if res == GameResult.X_WIN:
            reward = 1.0
        elif res == GameResult.DRAW:
            reward = 0.5
        else:
            reward = 0.0
        history["rewards"].append(reward)
        history["steps"].append(ep_steps)
        history["epsilons"].append(0.0)
        history["q_sizes"].append(0)
    return history


def plot_training_history(name: str, history: dict) -> Tuple[str, str, str, str]:
    """Create 4 light-theme figures (reward, steps, epsilon, q-size).
    Returns 4 base64 strings so UI can lay them out evenly.
    """
    rewards = history["rewards"]
    steps = history["steps"]
    epsilons = history["epsilons"]
    q_sizes = history["q_sizes"]
    eps = list(range(len(rewards)))
    # Light style to match white theme
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        plt.style.use("default")

    # Reward
    fig_r, ax_r = plt.subplots(1, 1, figsize=(6, 3.6))
    if len(rewards) >= 2:
        ax_r.plot(eps, rewards, color="#2563eb", alpha=0.6)
    elif len(rewards) == 1:
        ax_r.scatter([0], [rewards[0]], color="#2563eb", s=24)
    if len(rewards) >= 50:
        ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax_r.plot(range(49, len(rewards)), ma, color="#f59e0b", linewidth=2.0, label="MA-50")
        ax_r.legend()
    ax_r.set_title("Reward per Episode (MA-50)")
    ax_r.set_xlabel("Episode")
    ax_r.set_ylabel("Reward")
    ax_r.grid(True, alpha=0.3)
    b64_r = fig_to_base64(fig_r)

    # Steps
    fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 3.6))
    if len(steps) >= 2:
        ax_s.plot(eps, steps, color="#9333ea", alpha=0.6)
    elif len(steps) == 1:
        ax_s.scatter([0], [steps[0]], color="#9333ea", s=24)
    if len(steps) >= 50:
        ma2 = np.convolve(steps, np.ones(50)/50, mode='valid')
        ax_s.plot(range(49, len(steps)), ma2, color="#10b981", linewidth=2.0, label="MA-50")
        ax_s.legend()
    ax_s.set_title("Steps per Episode (MA-50)")
    ax_s.set_xlabel("Episode")
    ax_s.set_ylabel("Steps")
    ax_s.grid(True, alpha=0.3)
    b64_s = fig_to_base64(fig_s)

    # Epsilon
    fig_e, ax_e = plt.subplots(1, 1, figsize=(6, 3.6))
    ax_e.plot(eps, epsilons, color="#db2777")
    ax_e.set_title("Exploration Rate (epsilon)")
    ax_e.set_xlabel("Episode")
    ax_e.set_ylabel("Epsilon")
    ax_e.grid(True, alpha=0.3)
    b64_e = fig_to_base64(fig_e)

    # Q-size
    fig_q, ax_q = plt.subplots(1, 1, figsize=(6, 3.6))
    ax_q.plot(eps, q_sizes, color="#ef4444")
    ax_q.set_title("Q-table Size")
    ax_q.set_xlabel("Episode")
    ax_q.set_ylabel("Entries")
    ax_q.grid(True, alpha=0.3)
    b64_q = fig_to_base64(fig_q)

    return b64_r, b64_s, b64_e, b64_q


def agent_q_visuals(agent: BaseAgent) -> Tuple[str, str]:
    """Return (heatmap_b64, histogram_b64) visualizing learned Q-values for RL agents.
    For non-RL agents returns empty strings.
    """
    values = []
    heat = np.zeros((3, 3), dtype=float)
    cnt = np.zeros((3, 3), dtype=int)

    def push(action, val):
        r, c = action
        heat[r, c] += val
        cnt[r, c] += 1
        values.append(val)

    if hasattr(agent, "q_table") and isinstance(getattr(agent, "q_table"), dict) and not isinstance(agent, DoubleQLearningAgent):
        # Generic single Q-table (QL, SARSA, Expected SARSA, Monte Carlo, DP planner)
        for (state, action), q in getattr(agent, "q_table").items():
            push(action, q)
    elif isinstance(agent, (QLearningAgent, SARSAAgent, ExpectedSARSAAgent)):
        for (state, action), q in agent.q_table.items():
            push(action, q)
    elif isinstance(agent, DoubleQLearningAgent):
        for (state, action), q in agent.q_table1.items():
            push(action, q)
        for (state, action), q in agent.q_table2.items():
            push(action, q)
    else:
        return "", ""

    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.where(cnt > 0, heat / cnt, 0.0)

    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        plt.style.use("default")

    # Heatmap
    fig_h, ax_h = plt.subplots(1, 1, figsize=(4.8, 4.2))
    im = ax_h.imshow(avg, cmap="viridis", vmin=float(np.min(avg)), vmax=float(np.max(avg) if np.max(avg) > 0 else 1.0))
    for r in range(3):
        for c in range(3):
            ax_h.text(c, r, f"{avg[r,c]:.2f}", ha='center', va='center')
    ax_h.set_title(f"{agent.name}: Avg Q by Cell")
    ax_h.set_xticks(range(3)); ax_h.set_yticks(range(3))
    fig_h.colorbar(im, ax=ax_h, shrink=0.85)
    b64_h = fig_to_base64(fig_h)

    # Histogram
    fig_g, ax_g = plt.subplots(1, 1, figsize=(5.2, 3.6))
    if len(values) == 0:
        values = [0.0]
    ax_g.hist(values, bins=30, color="#2563eb", alpha=0.8)
    ax_g.set_title(f"{agent.name}: Q-value Distribution ({len(values)} entries)")
    ax_g.set_xlabel("Q value"); ax_g.set_ylabel("Frequency")
    ax_g.grid(True, alpha=0.3)
    b64_g = fig_to_base64(fig_g)

    return b64_h, b64_g


def plot_compare_overlay(histories: Dict[str, Dict]) -> Tuple[str, str, str]:
    """Overlay curves for Rewards, Steps, and Epsilon across agents.
    histories: {name: history_dict}
    Returns three base64 images.
    """
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        plt.style.use("default")

    colors = {
        "Q-Learning": "#2563eb",
        "SARSA": "#16a34a",
        "Expected SARSA": "#9333ea",
        "Double Q-Learning": "#ef4444",
        "Monte Carlo": "#f59e0b",
        "Dynamic Programming": "#0ea5e9",
    }

    # Rewards overlay (MA-50)
    fig_r, ax_r = plt.subplots(1, 1, figsize=(8.2, 4.6))
    for name, h in histories.items():
        r = h["rewards"]; x = list(range(len(r)))
        ax_r.plot(x, r, alpha=0.18, color=colors.get(name, None))
        if len(r) >= 50:
            ma = np.convolve(r, np.ones(50)/50, mode='valid')
            ax_r.plot(range(49, len(r)), ma, linewidth=2.2, label=name, color=colors.get(name, None))
        else:
            ax_r.plot(x, r, linewidth=1.4, label=name, color=colors.get(name, None))
    ax_r.set_title("Reward per Episode (MA-50)")
    ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Reward"); ax_r.grid(True, alpha=0.3)
    ax_r.legend()
    b64_ro = fig_to_base64(fig_r)

    # Steps overlay (MA-50)
    fig_s, ax_s = plt.subplots(1, 1, figsize=(8.2, 4.6))
    for name, h in histories.items():
        s = h["steps"]; x = list(range(len(s)))
        ax_s.plot(x, s, alpha=0.18, color=colors.get(name, None))
        if len(s) >= 50:
            ma2 = np.convolve(s, np.ones(50)/50, mode='valid')
            ax_s.plot(range(49, len(s)), ma2, linewidth=2.2, label=name, color=colors.get(name, None))
        else:
            ax_s.plot(x, s, linewidth=1.4, label=name, color=colors.get(name, None))
    ax_s.set_title("Steps per Episode (MA-50)")
    ax_s.set_xlabel("Episode"); ax_s.set_ylabel("Steps"); ax_s.grid(True, alpha=0.3)
    ax_s.legend()
    b64_so = fig_to_base64(fig_s)

    # Epsilon overlay
    fig_e, ax_e = plt.subplots(1, 1, figsize=(8.2, 4.6))
    for name, h in histories.items():
        e = h["epsilons"]; x = list(range(len(e)))
        ax_e.plot(x, e, linewidth=1.8, label=name, color=colors.get(name, None))
    ax_e.set_title("Exploration (epsilon)"); ax_e.set_xlabel("Episode"); ax_e.set_ylabel("Epsilon"); ax_e.grid(True, alpha=0.3)
    ax_e.legend()
    b64_eo = fig_to_base64(fig_e)

    return b64_ro, b64_so, b64_eo


# ------------------------ Flet UI ------------------------

def main(page: ft.Page):
    page.title = "Tic Tac Toe AI Arena (Flet)"
    # Switch to light theme as requested
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20

    # Safely schedule UI updates from worker threads across Flet versions
    def safe_ui(fn):
        if hasattr(page, "call_from_thread"):
            page.call_from_thread(fn)
        else:
            try:
                fn()
            except Exception:
                pass

    # Common state
    session_stats = {"games": 0, "human": 0, "ai": 0, "draws": 0}

    # -------------- Play tab --------------
    ai_dropdown = ft.Dropdown(
        label="AI Opponent",
        options=[ft.dropdown.Option(a) for a in ALL_AGENTS],
        value="Random",
        width=260,
    )
    status_text = ft.Text("Your turn (X)")
    # AI vs AI controls
    ai_vs_ai_cb = ft.Checkbox(label="Watch AI vs AI", value=False)
    agent_x_dd = ft.Dropdown(label="Agent X", options=[ft.dropdown.Option(a) for a in ALL_AGENTS], value="Minimax α-β", width=220)
    agent_o_dd = ft.Dropdown(label="Agent O", options=[ft.dropdown.Option(a) for a in ALL_AGENTS], value="Random", width=220)
    start_watch_btn = ft.ElevatedButton("Start AI Match")
    # Match stats table
    match_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Metric", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Value", weight=ft.FontWeight.W_600, color="#ffffff")),
        ],
        rows=[],
        heading_row_color="#2563eb",
        divider_thickness=0.7,
    )

    game = TicTacToe()
    human_player = Player.X
    ai_agent = create_agent_by_name(ai_dropdown.value, Player.O)

    def update_board_buttons():
        for r in range(3):
            for c in range(3):
                v = game.board[r, c]
                board_btns[r][c].text = "X" if v == 1 else ("O" if v == 2 else " ")
                # Light theme button colors
                blue = "#2563eb"   # blue-600
                red = "#dc2626"    # red-600
                empty = "#e5e7eb"  # gray-200
                txt_dark = "#111827"
                txt_light = "#ffffff"
                bg = blue if v == 1 else (red if v == 2 else empty)
                fg = txt_light if v in (1, 2) else txt_dark
                board_btns[r][c].style = ft.ButtonStyle(
                    bgcolor=bg,
                    color=fg,
                )
        page.update()

    def on_ai_changed(e):
        nonlocal ai_agent
        # Prefer a trained clone if available
        ai_agent = clone_trained_agent(ai_dropdown.value, Player.O) or create_agent_by_name(ai_dropdown.value, Player.O)
        new_game(None)

    ai_dropdown.on_change = on_ai_changed

    def make_ai_move():
        nonlocal ai_agent
        if ai_agent and game.check_winner() == GameResult.IN_PROGRESS and game.current_player == ai_agent.player:
            action = ai_agent.choose_action(game)
            game.make_move(action)

    def end_game_update(result: GameResult):
        if result == GameResult.X_WIN:
            status_text.value = "X Wins!"
            session_stats["human"] += 1
        elif result == GameResult.O_WIN:
            status_text.value = "O Wins!"
            session_stats["ai"] += 1
        else:
            status_text.value = "Draw!"
            session_stats["draws"] += 1
        session_stats["games"] += 1
        stats_label.value = (
            f"Games: {session_stats['games']}  |  Human Wins: {session_stats['human']}  |  "
            f"AI Wins: {session_stats['ai']}  |  Draws: {session_stats['draws']}\nCurrent AI: {ai_dropdown.value}"
        )
        # Populate match table for human vs AI as well
        winner = "X" if result == GameResult.X_WIN else ("O" if result == GameResult.O_WIN else "Draw")
        human_side = "X" if human_player == Player.X else "O"
        ai_side = "O" if human_player == Player.X else "X"
        moves = str(len(game.move_history))
        data = [
            ("Mode", "Human vs AI"),
            ("Human side", human_side),
            ("AI side", ai_side),
            ("AI agent", ai_dropdown.value),
            ("Winner", winner),
            ("Moves", moves),
        ]
        match_table.rows.clear()
        for idx, (k, v) in enumerate(data):
            bg = "#f8fafc" if idx % 2 == 0 else "#eef2ff"
            match_table.rows.append(
                ft.DataRow(cells=[ft.DataCell(ft.Text(k)), ft.DataCell(ft.Text(v))], color=bg)
            )

    def on_cell_click(r, c):
        # Disable human clicks when watching AI vs AI
        if ai_vs_ai_cb.value:
            return
        if game.check_winner() != GameResult.IN_PROGRESS:
            return
        if not game.is_valid_action((r, c)):
            return
        game.make_move((r, c))
        result = game.check_winner()
        if result != GameResult.IN_PROGRESS:
            end_game_update(result)
            update_board_buttons()
            return
        # AI turn
        make_ai_move()
        result = game.check_winner()
        if result != GameResult.IN_PROGRESS:
            end_game_update(result)
        update_board_buttons()

    def new_game(e):
        game.reset()
        status_text.value = "Your turn (X)"
        update_board_buttons()
        page.update()

    def start_ai_match(e):
        if not ai_vs_ai_cb.value:
            status_text.value = "Enable 'Watch AI vs AI' first"
            page.update()
            return
        start_watch_btn.disabled = True
        status_text.value = "AI vs AI running..."
        page.update()

        def worker():
            # instantiate agents
            ax = clone_trained_agent(agent_x_dd.value, Player.X) or create_agent_by_name(agent_x_dd.value, Player.X)
            ao = clone_trained_agent(agent_o_dd.value, Player.O) or create_agent_by_name(agent_o_dd.value, Player.O)
            game.reset()
            moves = 0

            while True:
                res = game.check_winner()
                if res != GameResult.IN_PROGRESS:
                    break
                cur = game.current_player
                ag = ax if cur == Player.X else ao
                action = ag.choose_action(game)
                game.make_move(action)
                moves += 1
                def upd():
                    update_board_buttons()
                    page.update()
                safe_ui(upd)
                time.sleep(0.4)

            result = game.check_winner()
            winner = "X" if result == GameResult.X_WIN else ("O" if result == GameResult.O_WIN else "Draw")

            def finish():
                status_text.value = f"AI match finished: {winner}"
                # fill table
                match_table.rows.clear()
                data = [
                    ("Agent X", ax.name),
                    ("Agent O", ao.name),
                    ("Winner", winner),
                    ("Moves", str(moves)),
                ]
                for idx, (k, v) in enumerate(data):
                    bg = "#f8fafc" if idx % 2 == 0 else "#eef2ff"
                    match_table.rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(k)), ft.DataCell(ft.Text(v))], color=bg))
                start_watch_btn.disabled = False
                page.update()

            safe_ui(finish)

        threading.Thread(target=worker, daemon=True).start()

    start_watch_btn.on_click = start_ai_match

    board_btns: List[List[ft.ElevatedButton]] = []
    board_rows: List[ft.Row] = []
    for r in range(3):
        row_btns = []
        for c in range(3):
            btn = ft.ElevatedButton(" ", width=80, height=80, on_click=lambda e, rr=r, cc=c: on_cell_click(rr, cc))
            row_btns.append(btn)
        board_btns.append(row_btns)
        board_rows.append(ft.Row(row_btns, alignment=ft.MainAxisAlignment.CENTER))

    stats_label = stats_text("Games: 0  |  Human Wins: 0  |  AI Wins: 0  |  Draws: 0\nCurrent AI: Random")

    play_tab = ft.Column(
        [
            ft.Row([ai_dropdown, ft.ElevatedButton("New Game", on_click=new_game)], spacing=20),
            ft.Row([ai_vs_ai_cb, agent_x_dd, agent_o_dd, start_watch_btn], spacing=12),
            ft.Column(board_rows, alignment=ft.MainAxisAlignment.CENTER),
            status_text,
            stats_label,
            card("Match statistics ", [match_table]),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=16,
        scroll=ft.ScrollMode.ALWAYS,
    )

    update_board_buttons()
    
    # Responsive sizing: resize board buttons based on window width
    def on_resize(e):
        try:
            w = getattr(page, "window_width", None) or page.width or 800
        except Exception:
            w = 800
        size = max(56, min(140, int(w / 12)))
        for r in range(3):
            for c in range(3):
                board_btns[r][c].width = size
                board_btns[r][c].height = size
        page.update()

    page.on_resize = on_resize

    # -------------- Training tab --------------
    train_agent_dd = ft.Dropdown(
        label="RL Agent",
        options=[ft.dropdown.Option(a) for a in TRAINABLE],
        value="Q-Learning",
        width=260,
    )
    # Opponent mode: self-play or specific agent
    train_mode_dd = ft.Dropdown(label="Opponent mode", width=220,
                                options=[ft.dropdown.Option("Self-play (same agent)"), ft.dropdown.Option("Against agent")],
                                value="Self-play (same agent)")
    opponent_dd = ft.Dropdown(label="Opponent agent", options=[ft.dropdown.Option(a) for a in ALL_AGENTS], value="Random", width=240, visible=False)
    # Opponent pretraining option
    opp_pretrain_cb = ft.Checkbox(label="Pretrain opponent", value=False)
    opp_pretrain_eps_tf = ft.TextField(label="Opponent pretrain eps", value="1000", width=180, visible=False,
                                       keyboard_type=ft.KeyboardType.NUMBER, input_filter=ft.NumbersOnlyInputFilter())
    episodes_tf = ft.TextField(label="Episodes", value=str(int(CFG.get("training_episodes", 2000))), width=150)
    start_train_btn = ft.ElevatedButton("Start Training")
    stop_train_btn = ft.ElevatedButton("Stop", disabled=True)
    train_prog = ft.ProgressBar(width=400, value=0)
    train_status = subheading("Idle")
    # Four equally sized images for even layout (bigger for readability)
    img_reward = ft.Image(height=360, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    img_steps = ft.Image(height=360, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    img_epsilon = ft.Image(height=360, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    img_qsize = ft.Image(height=360, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    # Q-table visuals (heatmap + histogram)
    img_qheat = ft.Image(height=360, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    img_qhist = ft.Image(height=360, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    # Stats panel and table
    stats_panel = stats_text("")
    stats_panel.visible = False
    train_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Metric", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Value", weight=ft.FontWeight.W_600, color="#ffffff")),
        ],
        rows=[],
        heading_row_color="#2563eb",
        divider_thickness=0.7,
    )

    def on_train_mode_change(e):
        opponent_dd.visible = train_mode_dd.value == "Against agent"
        opp_pretrain_eps_tf.visible = opp_pretrain_cb.value and opponent_dd.visible
        page.update()

    train_mode_dd.on_change = on_train_mode_change

    # Suggest faster defaults for different agents
    def on_train_agent_changed(e):
        # Do not override user's chosen episode count; leave field unchanged.
        page.update()

    train_agent_dd.on_change = on_train_agent_changed

    def on_opp_pretrain_toggle(e):
        opp_pretrain_eps_tf.visible = opp_pretrain_cb.value and (train_mode_dd.value == "Against agent")
        page.update()

    opp_pretrain_cb.on_change = on_opp_pretrain_toggle

    # training cancel flag
    training_stop = threading.Event()

    def do_training():
        start_train_btn.disabled = True
        stop_train_btn.disabled = False
        training_stop.clear()
        page.update()
        try:
            eps = int(episodes_tf.value)
        except Exception:
            eps = 2000
        name = train_agent_dd.value
        # Create agent as X for training (treat MC/DP as trainable variants in UI)
        if name == "Expected SARSA":
            agent = ExpectedSARSAAgent(Player.X)
        elif name == "SARSA":
            agent = SARSAAgent(Player.X)
        elif name == "Double Q-Learning":
            agent = DoubleQLearningAgent(Player.X)
        elif name == "Monte Carlo":
            agent = MonteCarloAgent(Player.X)
        elif name == "Dynamic Programming":
            agent = DynamicProgrammingAgent(Player.X)
        else:
            agent = QLearningAgent(Player.X)

        # Apply config hyperparameters to the agent if supported
        if hasattr(agent, "alpha"):
            try:
                agent.alpha = float(CFG.get("learning_rate", agent.alpha))
            except Exception:
                pass
        if hasattr(agent, "gamma"):
            try:
                agent.gamma = float(CFG.get("gamma", getattr(agent, "gamma", 0.95)))
            except Exception:
                pass

        # Configure opponent based on mode
        if train_mode_dd.value == "Against agent":
            opp = create_agent_by_name(opponent_dd.value, Player.O)
        else:
            # self-play = same algorithm as trainee
            opp = create_agent_by_name(name, Player.O)

        # Optional opponent pretraining
        if opp_pretrain_cb.value and train_mode_dd.value == "Against agent":
            try:
                pre_eps = int(opp_pretrain_eps_tf.value)
            except Exception:
                pre_eps = 1000
            if isinstance(opp, (QLearningAgent, SARSAAgent, ExpectedSARSAAgent, DoubleQLearningAgent)):
                opp.train(pre_eps)

        train_status.value = "Training... (no live plots)"
        train_prog.value = None  # indeterminate
        page.update()

        # Run training in background to avoid blocking UI
        def worker():
            t0 = time.perf_counter()
            # Choose appropriate history collection: prefer agent-provided training with history
            if hasattr(agent, "train_with_history"):
                try:
                    # Pass epsilon_decay from config for MC; DP will ignore extra kw and fall back below
                    history = agent.train_with_history(episodes=eps, opp_agent=opp, stop_event=training_stop, epsilon_decay=float(CFG.get("epsilon_decay", 0.995)))
                except TypeError:
                    # Fallback if signature differs
                    history = agent.train_with_history(episodes=eps)
            elif isinstance(agent, (QLearningAgent, SARSAAgent, ExpectedSARSAAgent, DoubleQLearningAgent)):
                history = train_agent_collect_history(
                    agent,
                    episodes=eps,
                    opp_agent=opp,
                    stop_event=training_stop,
                    epsilon_decay=float(CFG.get("epsilon_decay", 0.995)),
                )
            else:
                history = eval_agent_collect_history(agent, episodes=eps, opp_agent=opp)
            t1 = time.perf_counter()
            # Save and render plots
            b64_r, b64_s, b64_e, b64_q = plot_training_history(name, history)
            # Q-table visuals for RL agents
            heat_b64, hist_b64 = agent_q_visuals(agent)

            # Build stats
            last100 = max(1, min(100, len(history["rewards"])))
            avg_r_last = float(np.mean(history["rewards"][-last100:]))
            avg_s_last = float(np.mean(history["steps"][-last100:]))
            final_eps = history["epsilons"][-1] if history["epsilons"] else 0.0
            final_q = history["q_sizes"][-1] if history["q_sizes"] else 0
            # best moving average (50)
            if len(history["rewards"]) >= 50:
                ma = np.convolve(history["rewards"], np.ones(50)/50, mode='valid')
                best_ma = float(np.max(ma))
            else:
                best_ma = float(np.mean(history["rewards"]))

            def finish_ui():
                train_prog.value = 1
                train_status.value = "Training complete"
                img_reward.visible = True
                img_steps.visible = True
                img_epsilon.visible = True
                img_qsize.visible = True
                img_reward.src_base64 = b64_r
                img_steps.src_base64 = b64_s
                img_epsilon.src_base64 = b64_e
                img_qsize.src_base64 = b64_q
                if heat_b64:
                    img_qheat.visible = True
                    img_qhist.visible = True
                    img_qheat.src_base64 = heat_b64
                    img_qhist.src_base64 = hist_b64
                stats_panel.visible = True
                stats_panel.value = (
                    f"Episodes: {len(history['rewards'])}  |  Duration: {t1 - t0:.1f}s\n"
                    f"Final epsilon: {final_eps:.3f}  |  Q-table size: {final_q}\n"
                    f"Avg reward (last {last100}): {avg_r_last:.3f}  |  Avg steps (last {last100}): {avg_s_last:.2f}  |  Best MA50: {best_ma:.3f}"
                )
                # Fill training stats table (zebra rows)
                train_table.rows.clear()
                entries = [
                    ("Agent", name),
                    ("Episodes", str(len(history['rewards']))),
                    ("Duration (s)", f"{t1 - t0:.1f}"),
                    ("Final epsilon", f"{final_eps:.3f}"),
                    ("Q-table size", str(final_q)),
                    ("Avg reward (last %d)" % last100, f"{avg_r_last:.3f}"),
                    ("Avg steps (last %d)" % last100, f"{avg_s_last:.2f}"),
                    ("Best MA-50", f"{best_ma:.3f}"),
                ]
                for idx, (k, v) in enumerate(entries):
                    bg = "#f8fafc" if idx % 2 == 0 else "#eef2ff"
                    train_table.rows.append(ft.DataRow(cells=[ft.DataCell(ft.Text(k)), ft.DataCell(ft.Text(v))], color=bg))
                # Save trained model into session registry for reuse
                TRAINED_MODELS[name] = agent
                start_train_btn.disabled = False
                stop_train_btn.disabled = True
                page.update()

            safe_ui(finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    start_train_btn.on_click = lambda e: do_training()

    def stop_training(e):
        training_stop.set()
        train_status.value = "Stopping..."
        page.update()

    stop_train_btn.on_click = stop_training

    # Layout: scrolling column, 2x2 grid for images
    training_tab = ft.Column(
        [
            ft.Row([train_agent_dd, train_mode_dd, opponent_dd, episodes_tf, opp_pretrain_cb, opp_pretrain_eps_tf, start_train_btn, stop_train_btn], spacing=12),
            train_status,
            ft.Container(train_prog, padding=ft.padding.only(top=6, bottom=10)),
            card("Visualizations (rendered when training finishes)", [
                ft.Row([img_reward, img_steps], spacing=12),
                ft.Row([img_epsilon, img_qsize], spacing=12),
            ]),
            card("Learned values (RL agents)", [
                ft.Row([img_qheat, img_qhist], spacing=12)
            ]),
            # Remove extra text block before table as requested; show only the table
            card("Training statistics", [
                train_table,
            ]),
        ],
        spacing=16,
        scroll=ft.ScrollMode.ALWAYS,
    )

    # -------------- Tournament tab --------------
    # numeric field for games per matchup
    games_tf = ft.TextField(label="Games per matchup", value="50", width=200,
                            keyboard_type=ft.KeyboardType.NUMBER,
                            input_filter=ft.NumbersOnlyInputFilter())
    pretrain_cb = ft.Checkbox(label=f"Train RL agents first ({int(CFG.get('training_episodes', 2000))} eps)", value=True)
    use_trained_cb = ft.Checkbox(label="Use session trained models if available", value=True)
    # checkboxes to select which agents participate
    agent_checks = {name: ft.Checkbox(label=name, value=True) for name in ALL_AGENTS}
    # Layout checkboxes in rows (fallback for Flet versions without Wrap)
    agent_cb_list = list(agent_checks.values())
    rows = []
    for i in range(0, len(agent_cb_list), 3):
        rows.append(ft.Row(agent_cb_list[i:i+3], spacing=12))
    agent_checks_view = ft.Column(rows, spacing=6)
    start_tour_btn = ft.ElevatedButton("Start Tournament")
    tour_prog = ft.ProgressBar(width=400, value=0)
    tour_status = ft.Text("Idle")
    table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Agent", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Wins", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Draws", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Losses", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Win Rate", weight=ft.FontWeight.W_600, color="#ffffff")),
        ],
        rows=[],
        heading_row_color="#0ea5e9",
        divider_thickness=0.7,
    )
    img_tour_res = ft.Image(height=420, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    rl_vis_col = ft.Column([], spacing=12)

    def start_tournament(e):
        start_tour_btn.disabled = True
        tour_prog.value = None
        tour_status.value = "Running tournament..."
        page.update()

        def worker():
            selected = [n for n, cb in agent_checks.items() if cb.value]
            if len(selected) < 2:
                def warn():
                    tour_status.value = "Pick at least two agents."
                    tour_prog.value = 0
                    start_tour_btn.disabled = False
                    page.update()
                safe_ui(warn)
                return

            # Create agent instances (clone trained models when requested)
            agents = []
            for n in selected:
                if use_trained_cb.value:
                    clone = clone_trained_agent(n, Player.X)
                    if clone is not None:
                        agents.append(clone)
                        continue
                agents.append(create_agent_by_name(n, Player.X))

            if pretrain_cb.value:
                for ag in agents:
                    # Prefer agent-provided history training when available (e.g., DP/MC)
                    if hasattr(ag, "train_with_history"):
                        try:
                            ag.train_with_history(episodes=int(CFG.get("training_episodes", 2000)))
                        except TypeError:
                            ag.train_with_history(episodes=int(CFG.get("training_episodes", 2000)))
                    elif isinstance(ag, (QLearningAgent, SARSAAgent, ExpectedSARSAAgent, DoubleQLearningAgent)):
                        ag.train(int(CFG.get("training_episodes", 2000)))

            tour = Tournament(agents)
            try:
                games = int(games_tf.value)
            except Exception:
                games = 50
            # run round robin; progress updates are coarse here
            total_matches = len(agents) * (len(agents) - 1) // 2
            done = 0
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    tour.run_match(agents[i], agents[j], num_games=games)
                    done += 1
                    p = done / total_matches
                    def upd_progress(val=p):
                        tour_prog.value = val
                        tour_status.value = f"Progress: {int(val*100)}%"
                        page.update()
                    safe_ui(upd_progress)

            # Build in-memory visualization (based on Tournament.visualize_results)
            leaderboard = tour.get_leaderboard()
            if leaderboard:
                try:
                    plt.style.use("seaborn-v0_8")
                except Exception:
                    plt.style.use("default")
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('Tournament Results', fontsize=16, fontweight='bold')

                agents_names = [item[0] for item in leaderboard]
                win_rates = [item[1]['win_rate'] * 100 for item in leaderboard]
                colors = plt.cm.viridis(np.linspace(0, 1, len(agents_names)))
                bars = ax1.barh(agents_names, win_rates, color=colors)
                ax1.set_xlabel('Win Rate (%)'); ax1.set_title('Win Rate Comparison', fontsize=14, fontweight='bold'); ax1.set_xlim(0, 100)
                for bar, rate in zip(bars, win_rates):
                    width = bar.get_width(); ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f'{rate:.1f}%', va='center', fontsize=10)

                wins = [item[1]['wins'] for item in leaderboard]
                losses = [item[1]['losses'] for item in leaderboard]
                draws = [item[1]['draws'] for item in leaderboard]
                x = np.arange(len(agents_names)); width = 0.25
                ax2.bar(x - width, wins, width, label='Wins', color='#2ecc71')
                ax2.bar(x, draws, width, label='Draws', color='#f39c12')
                ax2.bar(x + width, losses, width, label='Losses', color='#e74c3c')
                ax2.set_ylabel('Count'); ax2.set_title('Win/Draw/Loss Distribution', fontsize=14, fontweight='bold')
                ax2.set_xticks(x); ax2.set_xticklabels(agents_names, rotation=45, ha='right'); ax2.legend(); ax2.grid(axis='y', alpha=0.3)

                n_agents = len(tour.agents)
                h2h_matrix = np.zeros((n_agents, n_agents))
                agent_names_all = [agent.name for agent in tour.agents]
                for i, agent1 in enumerate(tour.agents):
                    for j, agent2 in enumerate(tour.agents):
                        if i != j:
                            key1 = f"{agent1.name} vs {agent2.name}"
                            key2 = f"{agent2.name} vs {agent1.name}"
                            if key1 in tour.matchups:
                                stats = tour.matchups[key1]
                                w = stats['agent1_wins']; total = stats['games']
                                h2h_matrix[i, j] = (w / total * 100) if total > 0 else 0
                            elif key2 in tour.matchups:
                                stats = tour.matchups[key2]
                                w = stats['agent2_wins']; total = stats['games']
                                h2h_matrix[i, j] = (w / total * 100) if total > 0 else 0
                sns.heatmap(h2h_matrix, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=agent_names_all, yticklabels=agent_names_all, ax=ax3, cbar_kws={'label': 'Win Rate (%)'}, vmin=0, vmax=100)
                ax3.set_title('Head-to-Head Win Rates (%)'); ax3.set_xlabel('Opponent'); ax3.set_ylabel('Agent')

                total_games = [item[1]['games'] for item in leaderboard]
                ax4.pie(total_games, labels=agents_names, autopct='%1.1f%%', startangle=90, colors=colors)
                ax4.set_title('Games Played Distribution')
                img_b64 = fig_to_base64(fig)
            else:
                img_b64 = ""

            leaderboard = tour.get_leaderboard()

            def finish_ui():
                # Fill table
                table.rows.clear()
                for idx, (name, stats) in enumerate(leaderboard):
                    bg = "#ecfeff" if idx % 2 == 0 else "#e0f2fe"
                    table.rows.append(ft.DataRow(cells=[
                        ft.DataCell(ft.Text(name)),
                        ft.DataCell(ft.Text(str(stats['wins']))),
                        ft.DataCell(ft.Text(str(stats['draws']))),
                        ft.DataCell(ft.Text(str(stats['losses']))),
                        ft.DataCell(ft.Text(f"{stats['win_rate']*100:.1f}%")),
                    ], color=bg))
                tour_prog.value = 1
                tour_status.value = "Done"
                img_tour_res.visible = True
                img_tour_res.src_base64 = img_b64
                # RL agents' learned visuals
                rl_vis_col.controls.clear()
                for ag in agents:
                    if isinstance(ag, (QLearningAgent, SARSAAgent, ExpectedSARSAAgent, DoubleQLearningAgent)):
                        h_b64, g_b64 = agent_q_visuals(ag)
                        if h_b64:
                            rl_vis_col.controls.append(ft.Text(f"{ag.name} learned values:"))
                            rl_vis_col.controls.append(ft.Row([ft.Image(src_base64=h_b64, height=360, expand=1, fit=ft.ImageFit.CONTAIN), ft.Image(src_base64=g_b64, height=360, expand=1, fit=ft.ImageFit.CONTAIN)], spacing=12))
                start_tour_btn.disabled = False
                page.update()

            safe_ui(finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    start_tour_btn.on_click = start_tournament

    tournament_tab = ft.Column(
        [
            ft.Row([games_tf, pretrain_cb, use_trained_cb, start_tour_btn], spacing=16),
            heading("Agents to include:"),
            agent_checks_view,
            tour_status,
            ft.Container(tour_prog, padding=ft.padding.only(bottom=6)),
            card("Leaderboard", [table]),
            card("Visualization", [img_tour_res]),
            card("RL agents learned values (if any)", [rl_vis_col]),
        ],
        spacing=16,
        scroll=ft.ScrollMode.ALWAYS,
    )

    # -------------- Compare Agents tab --------------
    cmp_eps_tf = ft.TextField(label="Episodes", value=str(int(CFG.get("training_episodes", 2000))), width=180, keyboard_type=ft.KeyboardType.NUMBER, input_filter=ft.NumbersOnlyInputFilter())
    cmp_games_tf = ft.TextField(label="Games per matchup", value="50", width=180, keyboard_type=ft.KeyboardType.NUMBER, input_filter=ft.NumbersOnlyInputFilter())
    cmp_selfplay_cb = ft.Checkbox(label="Include self-play matches", value=True)
    # For RL overlays we (re)train agents here to collect histories, independent from session registry.
    # We still evaluate all agents (RL and non-RL) in head-to-head matches below.
    # Select agents to compare (any of the 9)
    cmp_agent_checks = {name: ft.Checkbox(label=name, value=(name in ["Q-Learning", "SARSA", "Expected SARSA", "Double Q-Learning"])) for name in ALL_AGENTS}
    cmp_agent_rows = []
    cmp_list = list(cmp_agent_checks.values())
    for i in range(0, len(cmp_list), 3):
        cmp_agent_rows.append(ft.Row(cmp_list[i:i+3], spacing=12))
    cmp_agent_picker = ft.Column(cmp_agent_rows, spacing=6)

    cmp_start_btn = ft.ElevatedButton("Run Comparison")
    cmp_prog = ft.ProgressBar(width=400, value=0)
    cmp_status = ft.Text("Idle")
    cmp_img_reward = ft.Image(height=420, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    cmp_img_steps = ft.Image(height=420, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    cmp_img_eps = ft.Image(height=420, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    cmp_img_final = ft.Image(height=420, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    cmp_img_matrix = ft.Image(height=480, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    cmp_img_bars = ft.Image(height=420, fit=ft.ImageFit.CONTAIN, visible=False, expand=1)
    cmp_rl_toggle = ft.Checkbox(label="Show learned-value visuals (RL agents)", value=False)
    cmp_rl_vis_col = ft.Column([], spacing=12, visible=False)
    def on_cmp_rl_toggle(e):
        cmp_rl_vis_col.visible = cmp_rl_toggle.value
        page.update()
    cmp_rl_toggle.on_change = on_cmp_rl_toggle
    cmp_summary_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Agent", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("AvgR(last100)", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Best MA-50", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Final ε", weight=ft.FontWeight.W_600, color="#ffffff")),
            ft.DataColumn(ft.Text("Q-size", weight=ft.FontWeight.W_600, color="#ffffff")),
        ],
        rows=[],
        heading_row_color="#9333ea",
        divider_thickness=0.7,
    )

    def do_compare(e):
        cmp_start_btn.disabled = True
        cmp_status.value = "Running..."; cmp_prog.value = None
        page.update()

        def worker():
            try:
                eps = int(cmp_eps_tf.value)
            except Exception:
                eps = 3000

            selected = [n for n, cb in cmp_agent_checks.items() if cb.value]
            if not selected:
                def warn():
                    cmp_status.value = "Pick at least one agent."
                    cmp_prog.value = 0
                    cmp_start_btn.disabled = False
                    page.update()
                safe_ui(warn)
                return

            histories = {}
            agents_for_eval = []
            # Train or evaluate each selected agent to build overlays
            for name in selected:
                ag = create_agent_by_name(name, Player.X)
                if hasattr(ag, "train_with_history"):
                    try:
                        histories[name] = ag.train_with_history(episodes=eps)
                    except TypeError:
                        histories[name] = ag.train_with_history(episodes=eps)
                elif name in TRAINABLE:
                    histories[name] = train_agent_collect_history(ag, episodes=eps)
                else:
                    histories[name] = eval_agent_collect_history(ag, episodes=eps)
                agents_for_eval.append(ag)

            b64_r, b64_s, b64_e = plot_compare_overlay(histories)

            # Prepare RL agents learned-value visuals
            rl_vis_data = []  # list of (name, heat_b64, hist_b64)
            for ag in agents_for_eval:
                # Include MC and DP visuals as they expose q_table
                h_b64, g_b64 = agent_q_visuals(ag)
                if h_b64:
                    rl_vis_data.append((ag.name, h_b64, g_b64))

            # Final win rate vs Random (more RL metric)
            def eval_winrate(agent: BaseAgent, games: int = 300) -> float:
                game = TicTacToe()
                wins = 0; draws = 0
                opp = RandomAgent(Player.O)
                # Ensure agent plays X for evaluation simplicity
                agent.player = Player.X
                for _ in range(games):
                    game.reset()
                    while game.check_winner() == GameResult.IN_PROGRESS:
                        if game.current_player == Player.X:
                            a = agent.choose_action(game)
                        else:
                            a = opp.choose_action(game)
                        game.make_move(a)
                    res = game.check_winner()
                    if res == GameResult.X_WIN:
                        wins += 1
                    elif res == GameResult.DRAW:
                        draws += 1
                return (wins / games) * 100.0

            # Compute win rate vs Random for all selected
            win_rates = {ag.name: eval_winrate(ag, 300) for ag in agents_for_eval}

            # Head-to-head matches among selected agents (round robin)
            try:
                games = int(cmp_games_tf.value)
            except Exception:
                games = 50
            tour = Tournament(agents_for_eval)
            # Round-robin different types
            for i in range(len(agents_for_eval)):
                for j in range(i + 1, len(agents_for_eval)):
                    tour.run_match(agents_for_eval[i], agents_for_eval[j], num_games=games)
            # Optional self-play (same type vs same type)
            if cmp_selfplay_cb.value:
                for i in range(len(agents_for_eval)):
                    a1 = agents_for_eval[i]
                    # Deep clone the same agent to the O side, copying learned knowledge
                    a2 = clone_agent_instance(a1, Player.O)
                    tour.run_match(a1, a2, num_games=games)

            # Build compare visualizations: separate head-to-head heatmap and per-agent W/D/L bars
            leaderboard = tour.get_leaderboard()
            b64_h2h = ""; b64_wdl = ""
            if leaderboard:
                try:
                    plt.style.use("seaborn-v0_8")
                except Exception:
                    plt.style.use("default")
                n = len(agents_for_eval)
                h2h = np.zeros((n, n))
                labels = [a.name for a in agents_for_eval]
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            key = f"{labels[i]} vs {labels[j]}"
                            if key in tour.matchups:
                                stats = tour.matchups[key]
                                h2h[i, j] = (stats['agent1_wins'] / stats['games']) * 100 if stats['games'] else 0
                        else:
                            k1 = f"{labels[i]} vs {labels[j]}"; k2 = f"{labels[j]} vs {labels[i]}"
                            if k1 in tour.matchups:
                                st = tour.matchups[k1]; h2h[i, j] = (st['agent1_wins'] / st['games']) * 100 if st['games'] else 0
                            elif k2 in tour.matchups:
                                st = tour.matchups[k2]; h2h[i, j] = (st['agent2_wins'] / st['games']) * 100 if st['games'] else 0
                fig_h, ax_h = plt.subplots(1, 1, figsize=(8.5, 7.0))
                sns.heatmap(h2h, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=labels, yticklabels=labels, ax=ax_h, cbar_kws={'label': 'Win Rate (%)'}, vmin=0, vmax=100)
                ax_h.set_title('Head-to-Head Win Rates (%)')
                b64_h2h = fig_to_base64(fig_h)
                # W/D/L grouped bars
                names_bar = [item[0] for item in leaderboard]
                wins = [item[1]['wins'] for item in leaderboard]
                draws = [item[1]['draws'] for item in leaderboard]
                losses = [item[1]['losses'] for item in leaderboard]
                x = np.arange(len(names_bar)); width = 0.25
                fig_b, ax_b = plt.subplots(1, 1, figsize=(12.0, 6.0))
                ax_b.bar(x - width, wins, width, label='Wins', color='#22c55e')
                ax_b.bar(x, draws, width, label='Draws', color='#f59e0b')
                ax_b.bar(x + width, losses, width, label='Losses', color='#ef4444')
                ax_b.set_xticks(x); ax_b.set_xticklabels(names_bar, rotation=45, ha='right')
                ax_b.set_ylabel('Count'); ax_b.set_title('Per-agent Win/Draw/Loss')
                ax_b.grid(axis='y', alpha=0.3); ax_b.legend()
                b64_wdl = fig_to_base64(fig_b)
            try:
                plt.style.use("seaborn-v0_8")
            except Exception:
                plt.style.use("default")
            figf, axf = plt.subplots(1, 1, figsize=(8.2, 4.6))
            names = list(win_rates.keys()); vals = [win_rates[n] for n in names]
            cmap = plt.cm.viridis(np.linspace(0, 1, max(1, len(names))))
            axf.bar(names, vals, color=cmap)
            axf.set_ylim(0, 100); axf.set_ylabel("Win rate vs Random (%)"); axf.set_title("Final Evaluation")
            for i, v in enumerate(vals):
                axf.text(i, v + 1, f"{v:.1f}%", ha='center')
            b64_final = fig_to_base64(figf)

            # Prepare per-agent summaries from histories
            summaries = []
            for name in selected:
                h = histories.get(name, {"rewards": [], "steps": [], "epsilons": [], "q_sizes": []})
                rw = h.get("rewards", [])
                eps_arr = h.get("epsilons", [])
                qs = h.get("q_sizes", [])
                last_n = min(100, len(rw)) if len(rw) > 0 else 0
                avg_last = float(np.mean(rw[-last_n:])) if last_n > 0 else 0.0
                if len(rw) >= 50:
                    ma = np.convolve(rw, np.ones(50)/50, mode='valid')
                    best_ma = float(np.max(ma)) if len(ma) > 0 else (float(np.mean(rw)) if len(rw) > 0 else 0.0)
                else:
                    best_ma = float(np.mean(rw)) if len(rw) > 0 else 0.0
                final_eps = (eps_arr[-1] if len(eps_arr) > 0 else None)
                final_q = int(qs[-1]) if len(qs) > 0 else 0
                summaries.append((name, avg_last, best_ma, final_eps, final_q))

            def finish():
                cmp_prog.value = 1
                cmp_status.value = "Done"
                cmp_img_reward.visible = True; cmp_img_reward.src_base64 = b64_r
                cmp_img_steps.visible = True; cmp_img_steps.src_base64 = b64_s
                cmp_img_eps.visible = True; cmp_img_eps.src_base64 = b64_e
                cmp_img_final.visible = True; cmp_img_final.src_base64 = b64_final
                if b64_h2h:
                    cmp_img_matrix.visible = True; cmp_img_matrix.src_base64 = b64_h2h
                if b64_wdl:
                    cmp_img_bars.visible = True; cmp_img_bars.src_base64 = b64_wdl
                # Populate RL visuals section
                cmp_rl_vis_col.controls.clear()
                for (nm, hb, gb) in rl_vis_data:
                    cmp_rl_vis_col.controls.append(ft.Text(f"{nm} learned values:"))
                    cmp_rl_vis_col.controls.append(
                        ft.Row([
                            ft.Image(src_base64=hb, height=360, expand=1, fit=ft.ImageFit.CONTAIN),
                            ft.Image(src_base64=gb, height=360, expand=1, fit=ft.ImageFit.CONTAIN),
                        ], spacing=12)
                    )
                cmp_rl_vis_col.visible = cmp_rl_toggle.value
                # Fill summary table (zebra rows)
                cmp_summary_table.rows.clear()
                for idx, (nm, avg_last, best_ma, final_eps, final_q) in enumerate(summaries):
                    bg = "#faf5ff" if idx % 2 == 0 else "#ede9fe"
                    cmp_summary_table.rows.append(
                        ft.DataRow(cells=[
                            ft.DataCell(ft.Text(nm)),
                            ft.DataCell(ft.Text(f"{avg_last:.3f}")),
                            ft.DataCell(ft.Text(f"{best_ma:.3f}")),
                            ft.DataCell(ft.Text("-" if final_eps is None else f"{final_eps:.3f}")),
                            ft.DataCell(ft.Text(str(final_q))),
                        ], color=bg)
                    )
                cmp_start_btn.disabled = False
                page.update()

            safe_ui(finish)

        threading.Thread(target=worker, daemon=True).start()

    cmp_start_btn.on_click = do_compare

    compare_tab = ft.Column(
        [
            heading("Pick agents to compare:"),
            cmp_agent_picker,
            ft.Row([cmp_eps_tf, cmp_games_tf, cmp_selfplay_cb, cmp_start_btn], spacing=16),
            cmp_status,
            ft.Container(cmp_prog, padding=ft.padding.only(bottom=6)),
            card("Overlay curves (RL and eval variants)", [cmp_img_reward, cmp_img_steps, cmp_img_eps]),
            card("Final win-rate vs Random (%)", [cmp_img_final]),
            card("Head-to-head results among selected agents", [cmp_img_matrix]),
            card("Per-agent win/draw/loss", [cmp_img_bars]),
            cmp_rl_toggle,
            card("Learned-value visuals", [cmp_rl_vis_col]),
            card("Per-agent summary", [cmp_summary_table]),
        ],
        spacing=16,
        scroll=ft.ScrollMode.ALWAYS,
    )

    # -------------- Tabs wrapper --------------
    tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="Play vs AI", content=ft.Container(content=play_tab, padding=ft.padding.only(top=28, left=16, right=16, bottom=16))),
            ft.Tab(text="Training", content=ft.Container(content=training_tab, padding=ft.padding.only(top=28, left=16, right=16, bottom=16))),
            ft.Tab(text="Tournament", content=ft.Container(content=tournament_tab, padding=ft.padding.only(top=28, left=16, right=16, bottom=16))),
            ft.Tab(text="Compare", content=ft.Container(content=compare_tab, padding=ft.padding.only(top=28, left=16, right=16, bottom=16))),
        ],
        expand=1,
    )

    page.add(tabs)


if __name__ == "__main__":
    # Run as a desktop app (system port, fast dev)
    ft.app(target=main)
