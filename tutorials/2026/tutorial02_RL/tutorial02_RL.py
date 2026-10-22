# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 2: Introduction to Reinforcement Learning (RL)

    **Angela Radulescu**

    New York Computational Psychiatry Workshop · 2026
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a [marimo](https://marimo.io) notebook. Marimo is *reactive*: when you change a cell or move a slider, every cell that depends on it re-runs automatically, so we don't have to worry about running cells in the wrong order.

    Today, we will learn how to use reinforcement learning (RL) to simulate behavior in different environments (tasks). This tutorial is intended to get you to a point where you can implement RL algorithms for a task of your choice. As we go along, we will go over both formal definitions and code.

    Let us dive in! The cell below imports the packages we need.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 13})  # larger text in all plots
    return Rectangle, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is reinforcement learning?

    **Reinforcement learning (RL)** refers to a set of learning processes that allow agents to make adaptive decisions over time in order to achieve a **goal**. What makes RL different from other types of learning (e.g. supervised learning) is that the feedback used to change behavior (i.e., the **reinforcement**) can be:

    1. **Sparse** -- reinforcement doesn't have to occur often
    2. **Delayed** -- reinforcement can occur after many actions are taken
    3. **Uncertain** -- an action may not always result in reinforcement
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Foundations

    RL starts from the concept of an **agent** that is acting in an **environment**.

    You can think of RL as a sort of "physics of behavior." It is a set of mathematical formulas that describe how intelligent agents interact with the *external* world. And how that interaction changes their *internal* world (e.g, cognition, beliefs, goals).

    At any given **timepoint** $t$, the agent can be in one of many **states** $s_t \in S$ of the environment, and decide to take one of many **actions** $a_t \in A$. Taking a particular action might switch the agent from $s_t$ to a new state $s_{t+1}$. Once an action is taken, the environment also responds with a reward $r_{t+1}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.image(
                src=mo.notebook_location()
                / "public"
                / "rl_agent_environment_loop.png",
                width=650,
            ),
            mo.md(r"""
    In this tutorial, we assume the agent can see the state directly ($o_t = s_t$). The next two tutorials cover what changes when it can't, and the agent has to infer the state from noisy observations (the *State Inference* box).
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Environment

    How the environment reacts to the agent's actions is defined by a **model**, which the agent may or may not know. The model has two main components: the transition function $T$ and the reward function $R$.

    Let’s say when we are in state $s$, we decide to take action $a$ to arrive in the next state $s’$ and obtain reward $r$. We refer to this as a **transition** step, represented by a tuple $(s, a, s’, r)$. You can think of this tuple as a unit of experience that the agent gets to learn from.

    The (repeated) tuples the agent will experience depend on how $T$ and $R$ are defined.

    The transition function $T$ defines the probability of transitioning from state $s$ to state $s'$ after taking action $a$ while obtaining reward $r$ (here $\mathbb{P}$ stands for "probability"):

    $$
    P(s',r \mid s,a) = \mathbb{P}[S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a]
    $$

    The reward function $R$ predicts the next reward triggered by one action:

    $$
    R(s,a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum\limits_{r}r\sum\limits_{s' \in S}P(s',r \mid s,a)
    $$

    *Example*: let's unpack this some more using a simple "Corridor" environment depicted below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src=mo.notebook_location() / "public" / "corridor_env.png", width=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The "Corridor" environment has the following properties:

    The state space consists of 3 states. The agent starts in State $1$ (Start), and can reach one of two **terminal** states, State $0$ (Hole) and State $2$ (Goal). When the agent reaches a terminal state, the **episode** (defined as a sequence of experience tuples) ends.

    $$
    s \in S = \{0, 1, 2\}
    $$

    The action space consists of two actions available to the agent: Action $0$ (Left) and Action $1$ (Right).

    $$
    a \in A = \{0, 1\}
    $$

    The "Corridor" has a **deterministic** transition function: a *Left* action always moves the agent to the left, and a *Right* action always moves the agent to the right.

    The "Corridor" reward function gives a +1 when the agent moves into State $2$ (*Goal*) and 0 otherwise.

    A convenient way to represent the model of the "Corridor" environment is in table form:

    | State  | Action  | Next state  | Transition probability  | Reward  |
    |-----------|-----------|-----------|-----------|-----------|
    | 0 (Hole) | 0 (Left) | 0 (Hole) | 1  | 0 |
    | 0 (Hole) | 1 (Right) | 0 (Hole) | 1  | 0 |
    | 1 (Start) | 0 (Left) | 0 (Hole) | 1  | 0 |
    | 1 (Start) | 1 (Right) | 2 (Goal) | 1  | +1 |
    | 2 (Goal) | 0 (Left) | 2 (Goal) | 1  | 0 |
    | 2 (Goal) | 1 (Right) | 2 (Goal) | 1  | 0 |

    *Exercise*: Using the formula for the reward function above, compute the reward the agent would expect to get if it starts in the Start state and moves Right.

    Let us now code up this environment. To define different components of our simulation, we will make use of [object oriented programming (OOP)](https://realpython.com/python3-object-oriented-programming/). OOP is helpful because it lets us define general classes we can reuse across different simulations to create new instances of each component (e.g. different instances of an agent, environment, etc).
    """)
    return


@app.cell
def _(Rectangle, np, plt):
    class Corridor(object):
        """Class for the corridor environment."""

        def __init__(self):
            self.n_states = 3
            self.n_actions = 2
            self.states = np.arange(0, self.n_states)
            self.actions = np.arange(0, self.n_actions)
            self.goal_state = 2
            self.state_labels = ["Hole", "Start", "Goal"]

            self.state = 1  # begin in Start state
            self.t = 0  # reset timestep
            self.terminate = False

        def visualize(self):
            fig, ax = plt.subplots(1, 1, figsize=(self.n_states, 1))
            # shade the Goal state green
            ax.add_patch(Rectangle((self.goal_state, 0), 1, 1, color="#008610", alpha=0.1))
            # grid lines
            for x in range(self.n_states + 1):
                ax.plot([x, x], [0, 1], color="black", alpha=0.33, linestyle=":")
            for y in [0, 1]:
                ax.plot([0, self.n_states], [y, y], color="black", alpha=0.33, linestyle=":")
            ax.set_xticks(self.states + 0.5)
            ax.set_xticklabels([f"{s} ({l})" for s, l in zip(self.states, self.state_labels)])
            ax.set_yticks([])
            ax.set_xlabel("State", fontsize=15)
            return fig

        def step(self, action):
            # the step method takes as input an action and changes the environment
            # this is where we encode the environment's true model (T and R)

            assert self.state in self.states, "Invalid state"
            assert action in self.actions, "Invalid action"

            print("Current state: " + str(self.state))
            print("Action: " + ("left" if action == 0 else "right"))

            if self.state == 1:
                if action == 0:  # agent picks left
                    self.state = 0
                    reward = 0
                else:  # agent picks right
                    self.state = 2
                    reward = 1
            else:  # terminal states are absorbing
                reward = 0
            self.terminate = True

            print("+ " + str(reward) + " reward")
            print("New state: " + str(self.state))

            # increment timestep
            self.t = self.t + 1
            if self.terminate:
                print("End of episode")

            return self.state, reward

        def reset(self):
            # the reset method resets the environment to its start state
            self.state = 1
            self.t = 0
            self.terminate = False

    return (Corridor,)


@app.cell
def _(Corridor):
    # This line creates an instance of the Corridor class
    corridor = Corridor()
    corridor.visualize()
    return


@app.cell(hide_code=True)
def _(mo):
    corridor_action = mo.ui.radio(
        options={"0 (Left)": 0, "1 (Right)": 1}, value="1 (Right)", label="Action:", inline=True
    )
    corridor_action
    return (corridor_action,)


@app.cell
def _(Corridor, corridor_action):
    # Each time you pick an action above, this cell starts a new episode and steps through it
    _env = Corridor()
    new_state, reward = _env.step(corridor_action.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some transition functions are **probabilistic**, meaning that the same action can lead to different states with different probabilities.

    *Exercise*: How would you modify the Corridor environment's `step` method to encode a transition function in which an action works as intended 80% of the time, but 20% of the time the agent "slips" and an action takes the agent to the opposite state?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We may or may not know how the model works and this differentiates two types of RL:

    1.   **Model-based RL** - rely on the model of the environment to decide how to act; know the model in advance, or learn it from experience
    2.   **Model-free RL** - no dependency on the model during learning and action selection

    Together, these elements (state space $S$, action space $A$, transition function $T$ and reward function $R$) define a **Markov Decision Process (MDP)**. You can think of the MDP as an abstract definition of a particular **task** an agent has to solve.

    *Discussion question*: what are some examples of tasks that come to mind which can be thought of in this way?

    All states in an MDP have the “Markov” property, referring to the fact that the future only depends on the current state, not the history. Or, in other words, the future and the past are conditionally independent given the present, as the current state encapsulates all the statistics we need to decide the future. This is one of the main **assumptions** of RL (assumption = what we take to be true from the start).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Agent

    Another key assumption of RL is that an agent's objective is to maximize long term future reward. This future reward is known as **return**, and it is defined as the total sum of discounted rewards going forward:

    $$
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...  = \sum\limits_{k = 0}^{\infty} \gamma^k R_{t+k+1}
    $$

    The discount factor $\gamma \in [0, 1]$ downweights rewards in the future. It is there for mathematical convenience (with $\gamma < 1$ the infinite sum stays finite), but also has an interesting psychological interpretation -- we know humans tend to discount benefits that are not immediate.

    The agent's **behavior** is represented by a **policy** $\pi$ which encodes what action to take in each state $s$. You can think of this as a function that maps between states and actions. This mapping can be either deterministic (always take the same action in a state) or stochastic (take different actions with different probabilities):

    $$
    \text{Deterministic}: \pi(s) = a
    $$

    $$
    \text{Stochastic}: \pi(a \mid s) = \mathbb{P}_\pi[A_t = a \mid S_t = s]
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Value

    Most RL algorithms in the literature rely on the concept of **value**. Value measures the general "goodness" of a **state** or **state-action** pair: how good is it to be in a state, or how good is it to be in a state and do an action? You can think of these two types of value as loosely mapping onto Pavlovian and instrumental control. From this perspective, Pavlovian learning refers to learning reward predictions; instrumental learning refers to learning what to do, given reward predictions.

    The state-value of a state $s$ is the expected return if we are in this state at time $t$, $S_t = s$:

    $$
    V_{\pi}(s) =  \mathbb{E}_\pi[G_t \mid S_t = s]
    $$

    Similarly, we define the action-value ("Q-value") of a state-action pair as:

    $$
    Q_{\pi}(s,a) =  \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
    $$

    By definition, the optimal value function produces the maximum return:

    $$
    V_*(s) = \max\limits_{\pi}V_{\pi}(s)
    $$

    $$
    Q_*(s, a) = \max\limits_{\pi}Q_{\pi}(s, a)
    $$

    And the optimal policy is the one that achieves the optimal value functions; once we know $Q_*$, we can read it off by picking the best action in every state:

    $$
    \pi_*(s) = \underset{a}{\mathrm{argmax}} \ Q_*(s,a)
    $$

    Most RL algorithms you will encounter in the computational psychiatry literature assume reward maximization defined as above, and aim to have the agent (e.g. human participants, rodents, monkeys) learn the optimal value function and/or the optimal policy. While this discussion is beyond the scope of our tutorial, the limits of this assumption is a matter of active debate in cognitive science ([Juechems & Summerfield](https://arxiv.org/pdf/2407.10583)) and AI ([Silver et al.](https://www.sciencedirect.com/science/article/pii/S0004370221000862), [Abel et al.](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_89.pdf)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Bellman equation

    The **Bellman equation** expresses value recursively: the value of what you do now is the reward you get immediately, plus the (discounted) value of where you end up. For the optimal action values:

    $$
    Q_*(s,a) = R(s,a) + \gamma \sum\limits_{s' \in S} P_{s,s'}^a \max\limits_{a' \in A} Q_*(s',a')
    $$

    Unpacking the terms:

    * $Q_*(s,a)$ = the value of taking action $a$ in state $s$, and acting optimally afterwards
    * $R(s,a)$ = the immediate reward obtained after taking action $a$ in state $s$
    * $\gamma$ = discount factor
    * $P_{s,s'}^a$ = probability of ending up in $s'$ after taking action $a$ in $s$
    * $\max_{a'} Q_*(s',a')$ = the value of the best action available in the next state

    *Discussion question*: does this logic make intuitive sense? What does the $\max$ assume about how the agent will behave in the future?

    When the agent knows the model ($R$ and $P$) and the task is small, this equation can be solved directly (e.g., with **dynamic programming**) to find the optimal policy. This is useful for asking how optimal behavior *should* change under different assumptions about the problem. *Example paper*: [Zorowitz et al.](https://pubmed.ncbi.nlm.nih.gov/34036174/) used this approach to show how anxious symptomatology can result from relaxing the assumption that the agent makes the return-maximizing choice at each future step (the $\max$ above).

    Most of the time, though, neither the agent nor the participant knows the model. The algorithms below instead *learn* values from experience, one transition at a time. Notice that Q-Learning's update is just this equation, applied to a single sampled transition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: a pessimistic agent in an open field

    Let's compute the Bellman equation for the open-field task from [Zorowitz et al.](https://pubmed.ncbi.nlm.nih.gov/34036174/). The agent starts at the bottom of an 11 × 11 grid and can move up, down, left or right. Two terminal squares sit near the top: one gives a reward (+10), and the other gives a punishment (−10).

    The only change from the optimality equation above is the $\max$. A **pessimistic** agent with weight $w$ assumes that in the next state it will get a mix of its best and worst options:

    $$
    Q(s,a) = R(s,a) + \gamma \Big[\, w \max\limits_{a'} Q(s',a') + (1-w) \min\limits_{a'} Q(s',a') \Big]
    $$

    With $w = 1$ this is the optimal agent. As $w$ decreases, the agent increasingly expects its future self to make the worst possible move.

    Because the agent knows the model, we can solve the equation with **value iteration**: start with all $Q = 0$ and apply the equation to every state over and over until the values stop changing.
    """)
    return


@app.cell
def _(np):
    class OpenField(object):
        """Open field task from Zorowitz et al.

        States are squares of a size x size grid, numbered row by row.
        Actions are 0 = up, 1 = down, 2 = left, 3 = right. Moves that would
        leave the grid are not available. Entering a terminal square gives
        its reward and ends the episode.
        """

        def __init__(self, size=11, start=(10, 5), rewards={(1, 2): 10, (1, 8): -10}):
            self.size = size
            self.n_states = size * size
            self.start = start[0] * size + start[1]
            self.rewards = {r * size + c: v for (r, c), v in rewards.items()}
            self.terminal = np.array(list(self.rewards))

            # next_state[s, a] is where action a takes you from s (-1 if not allowed)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.next_state = np.full((self.n_states, 4), -1)
            for s in range(self.n_states):
                r, c = divmod(s, size)
                for a, (dr, dc) in enumerate(moves):
                    if 0 <= r + dr < size and 0 <= c + dc < size:
                        self.next_state[s, a] = (r + dr) * size + (c + dc)
            self.valid = self.next_state >= 0

            # R[s, a]: reward for moving into a terminal square
            self.R = np.zeros((self.n_states, 4))
            for s, v in self.rewards.items():
                self.R[self.next_state == s] = v
            self.R[~self.valid] = np.nan


    def value_iteration(env, w=1.0, gamma=0.95, tol=1e-6, max_iter=2000):
        """Solve the pessimistic Bellman equation by value iteration.

        Returns Q (n_states x 4) and V(s) = max_a Q(s, a).
        """
        Q = np.where(env.valid, 0.0, np.nan)
        next_state = np.where(env.valid, env.next_state, 0)

        for _ in range(max_iter):
            # value of each next state: mix of its best and worst action
            V_next = w * np.nanmax(Q, axis=1) + (1 - w) * np.nanmin(Q, axis=1)
            V_next[env.terminal] = 0  # the episode ends in terminal states

            Q_new = np.where(env.valid, env.R + gamma * V_next[next_state], np.nan)
            converged = np.nanmax(np.abs(Q_new - Q)) < tol
            Q = Q_new
            if converged:
                break

        V = np.nanmax(Q, axis=1)
        V[env.terminal] = 0
        return Q, V


    def greedy_path(env, Q):
        """Follow the highest-valued action from the start until a terminal state or a loop."""
        path = [env.start]
        while path[-1] not in env.terminal:
            s_next = env.next_state[path[-1], np.nanargmax(Q[path[-1]])]
            if s_next in path:
                break
            path.append(s_next)
        return path

    return OpenField, greedy_path, value_iteration


@app.function
def plot_open_field(env, V, path, title, ax):
    """Heatmap of state values with the greedy path drawn as arrows."""
    grid = V.reshape(env.size, env.size).copy()
    for s, v in env.rewards.items():
        grid[divmod(s, env.size)] = v  # color terminal squares by their outcome
    im = ax.imshow(grid, cmap="RdBu", vmin=-10, vmax=10)
    for s1, s2 in zip(path[:-1], path[1:]):
        (y1, x1), (y2, x2) = divmod(s1, env.size), divmod(s2, env.size)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    sy, sx = divmod(env.start, env.size)
    ax.scatter(sx, sy, s=80, color="black", zorder=3)
    for s, v in env.rewards.items():
        ry, rx = divmod(s, env.size)
        ax.text(rx, ry, f"{v:+d}", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    return im


@app.cell(hide_code=True)
def _(mo):
    zor_w = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="pessimism weight w", show_value=True)
    zor_gamma = mo.ui.slider(0.5, 0.99, step=0.01, value=0.95, label="discount γ", show_value=True)
    mo.vstack([
        mo.md("The three panels on the left use the values from the paper ($w$ = 1, 0.5, 0). The right panel uses the sliders."),
        zor_w,
        zor_gamma,
    ])
    return zor_gamma, zor_w


@app.cell
def _(OpenField, greedy_path, plt, value_iteration, zor_gamma, zor_w):
    open_field = OpenField()

    _panels = [(1.0, 0.95), (0.5, 0.95), (0.0, 0.95), (zor_w.value, zor_gamma.value)]
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for _ax, (_w, _g) in zip(_axes, _panels):
        _Q, _V = value_iteration(open_field, w=_w, gamma=_g)
        _im = plot_open_field(open_field, _V, greedy_path(open_field, _Q), f"w = {_w:g}, γ = {_g:g}", _ax)
    _axes[-1].set_title(f"your agent: w = {zor_w.value:g}, γ = {zor_gamma.value:g}", fontsize=14)
    _fig.colorbar(_im, ax=_axes, shrink=0.8, label="V(s)")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The color shows each square's value, and the arrows show the path the agent takes from the start (black dot).

    *Discussion question*: How does the path change as $w$ goes from 1 to 0? Why does the $w = 0$ agent never reach the reward, even though the reward is just as close as the punishment? What kind of behavior in anxiety might this resemble?

    *Exercise*: Move the reward or the punishment by changing `rewards` in `OpenField(...)`, e.g. `OpenField(rewards={(1, 2): 10, (3, 4): -10})`. How close can the punishment be before the optimal agent ($w = 1$) changes its route?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classic RL algorithms

    We will now discuss two of the most widely used RL algorithms in the literature: **Temporal-Difference Learning** and **Q-Learning**. While remarkably simple, these algorithms have been extensively studied because of the wealth of evidence suggesting that the brain is implementing this form of RL (for a detailed discussion, see this review by [Niv](https://www.sciencedirect.com/science/article/pii/S0022249608001181)).

    The table below maps out the algorithms you are most likely to encounter in computational psychiatry. Today we cover TD-Learning and Q-Learning in this section, and recap Rescorla–Wagner when we model behavior in the next one. The model-based rows compute values from a known model of the environment, which is what the open-field example above was doing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src=mo.notebook_location() / "public" / "rl_algorithms_table.png", width=750)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TD-Learning

    Temporal-Difference Learning is a general purpose algorithm for learning reward predictions. Unlike dynamic programming, it does not need to know the model: it learns from sampled experience. With enough experience (and a suitably decreasing learning rate) it converges to the true value function $V_\pi$ of whatever policy the agent is following. It embodies the key intuition of RL, which is using experience to update reward expectations.

    The update equation for TD-Learning is as follows:

    $$
    V(S_{t}) \leftarrow V(S_t) + \alpha[\underbrace{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}_{\delta_t \text{ (TD error)}}]
    $$

    Unpacking the terms:

    * $V(S_t)$ = value of current state
    * $V(S_{t+1})$ = value of future state
    * $0 < \alpha < 1$ = learning rate
    * $R_{t+1}$ = immediate reward
    * $\gamma$ = discount factor
    * $\delta_t$ = the TD error, or **reward prediction error (RPE)**: how much better or worse things turned out than expected
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src=mo.notebook_location() / "public" / "track_env.png", width=550)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us now implement TD-Learning for a simple environment. Consider a mouse running along a track until it reaches a terminal state where it encounters a piece of cheese. We can abstract this as a "single track" in which the agent doesn't take any actions, but simply moves forward on the track.
    """)
    return


@app.cell
def _(Rectangle, np, plt):
    class SingleTrack(object):
        """Class for the single track environment.

        Parameters
        ----------
        n_states : int
            Number of states on the track. States are numbered 1..n_states;
            the agent starts in state 1 and the reward is in state n_states.
        reward : float
            Reward delivered on entering the last state.
        """

        def __init__(self, n_states, reward=10):
            self.n_states = n_states
            self.states = np.arange(1, n_states + 1)
            self.reward_location = self.n_states
            self.reward = reward

            self.state = 1  # begin in Start state
            self.t = 0  # reset timestep
            self.terminate = False

        def visualize(self):
            fig, ax = plt.subplots(1, 1, figsize=(self.n_states, 1))
            # shade the reward state
            ax.add_patch(
                Rectangle((self.reward_location - 1, 0), 1, 1, color="#f9b02e", alpha=0.8)
            )
            # grid lines
            for x in range(self.n_states + 1):
                ax.plot([x, x], [0, 1], color="black", alpha=0.33, linestyle=":")
            for y in [0, 1]:
                ax.plot([0, self.n_states], [y, y], color="black", alpha=0.33, linestyle=":")
            ax.set_xticks(np.arange(self.n_states) + 0.5)
            ax.set_xticklabels(self.states)
            ax.set_yticks([])
            ax.set_xlabel("State", fontsize=15)
            return fig

        def step(self, verbose=False):
            if verbose:
                print("Current state: " + str(self.state))
                print("Moving right...")
            self.state = self.state + 1

            if self.state == self.reward_location:
                reward = self.reward
                self.terminate = True
            else:
                reward = 0

            if verbose:
                print("New state: " + str(self.state))

            # increment timestep
            self.t = self.t + 1

            return self.state, reward

        def reset(self):
            # the reset method resets the environment to its start state
            self.state = 1
            self.t = 0
            self.terminate = False

    return (SingleTrack,)


@app.cell
def _(SingleTrack):
    track = SingleTrack(10)
    track.visualize()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Time to code up our first agent!
    """)
    return


@app.cell
def _(np):
    class TDRL(object):
        """Class for the TDRL reward prediction algorithm.

        Parameters
        ----------
        alpha : float, range (0, 1)
            Learning rate.
        gamma : float, range (0, 1)
            Discount factor.
        """

        def __init__(self, env, alpha, gamma, v_init=0.0):
            self.a = alpha
            self.g = gamma

            # initialize values (index s-1 holds the value of state s)
            self.v = np.ones((env.n_states,)) * v_init
            # the terminal state has no future reward, so its value stays 0
            self.v[env.n_states - 1] = 0

            # most recent td-error experienced in each state
            self.d = np.zeros((env.n_states,))

        def update(self, current_state, new_state, reward, verbose=False):
            td_error = reward + self.g * self.v[new_state - 1] - self.v[current_state - 1]

            self.d[current_state - 1] = td_error
            self.v[current_state - 1] = self.v[current_state - 1] + self.a * td_error

            if verbose:
                print("delta: " + str(td_error))
                print("values: " + str(self.v))

    return (TDRL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is often convenient to write a simulation function that has the agent and environment interact for a fixed number of trials or episodes.
    """)
    return


@app.cell
def _(SingleTrack, TDRL, np):
    def run_simulation(params, n_episodes, omit_reward_on=None):
        """Run TDRL in the single track environment.

        omit_reward_on : int or None
            If given, the reward is withheld on this episode (0-indexed).
        """

        # make environment
        env = SingleTrack(params["n_states"])

        # initialize agent
        agent = TDRL(env, params["alpha"], params["gamma"])

        # initialize output containers: one row per episode, one column per state
        V = np.zeros((n_episodes, params["n_states"]))
        D = np.zeros((n_episodes, params["n_states"]))

        for e in range(n_episodes):
            env.reset()

            while not env.terminate:
                # step through environment
                current_state = env.state
                new_state, reward = env.step()

                if e == omit_reward_on:
                    reward = 0

                # update agent
                agent.update(current_state, new_state, reward)

            # record values and TD errors at the end of the episode
            V[e, :] = agent.v
            D[e, :] = agent.d

        return env, V, D

    return (run_simulation,)


@app.cell(hide_code=True)
def _(mo):
    td_alpha = mo.ui.slider(0.01, 1.0, step=0.01, value=0.2, label="learning rate α", show_value=True)
    td_gamma = mo.ui.slider(0.0, 1.0, step=0.01, value=0.9, label="discount γ", show_value=True)
    td_episodes = mo.ui.slider(1, 300, step=1, value=100, label="episodes", show_value=True)
    td_omit = mo.ui.checkbox(label="omit reward on the last episode")
    mo.vstack(
        [mo.md("**Try it:** move the sliders and watch the plots below update."), td_alpha, td_gamma, td_episodes, td_omit]
    )
    return td_alpha, td_episodes, td_gamma, td_omit


@app.cell
def _(run_simulation, td_alpha, td_episodes, td_gamma, td_omit):
    n_episodes = td_episodes.value

    params = {
        "n_states": 10,
        "alpha": td_alpha.value,
        "gamma": td_gamma.value,
    }

    track_env, V, D = run_simulation(
        params, n_episodes, omit_reward_on=n_episodes - 1 if td_omit.value else None
    )
    return D, V, n_episodes, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To visualize our results, we can plot the value function and the TD error (reward prediction error) for each state. Each line is one episode; darker lines are later episodes.
    """)
    return


@app.cell
def _(D, V, n_episodes, np, params, plt):
    _states = np.arange(1, params["n_states"] + 1)
    _labels = [str(s) for s in _states[:-1]] + ["R"]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    for _ax, _data, _cmap, _ylabel in [
        (_axes[0], V, plt.cm.Blues, "Value"),
        (_axes[1], D, plt.cm.Greens, "Reward prediction error"),
    ]:
        for _e in range(n_episodes):
            _ax.plot(
                _states[:-1], _data[_e, :-1], marker="o", markersize=3,
                color=_cmap(0.2 + 0.8 * _e / max(n_episodes - 1, 1)),
            )
        _ax.axvline(x=params["n_states"], color=_cmap(0.9), linewidth=4)
        _ax.set_xticks(_states)
        _ax.set_xticklabels(_labels)
        _ax.set_xlabel("State", fontsize=15)
        _ax.set_ylabel(_ylabel, fontsize=15)
        _ax.spines[["top", "right"]].set_visible(False)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One nice intuition for what TD-Learning does is “backing up errors”.
    Each line on the value plot shows you what happens to the value of states preceding the reward when we apply TD-Learning over many episodes (darker lines mean later episodes).

    In early episodes, only the value of states immediately before the reward becomes positive (because that state predicts the imminent reward).

    But as the agent experiences this track more and more times, the error gets propagated backwards to earlier states in the track — they each acquire a bit of predictive value.

    This is exactly how we think conditioning works: if you walk multiple times to your favorite restaurant and take the same route, in time, you will learn to predict that different parts of your route (like a park or a street corner) lead to reward.

    *Discussion question*: What happens to the reward prediction error as the agent experiences the track multiple times? Why? What happens to the value of early states when you lower $\gamma$?

    *Exercise*: Suppose reward was omitted on a random episode during the task. What do you think would happen to the RPE when the agent reaches the last state during that episode? Make a prediction, then tick the "omit reward" box above to check it. This is exactly the experiment that led to the **[reward prediction error hypothesis](https://www.science.org/doi/10.1126/science.275.5306.1593)** of dopamine function. For a current view of this topic, see this [review](https://www.nature.com/articles/s41593-024-01705-4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q-Learning

    The TD-Learning algorithm tells us how to update our predictions, but does not tell us how to act. We can extend the logic of learning from temporal differences to learn **action values** $Q(s, a)$ instead of state values.

    **Q-Learning** does this with one twist: when it looks ahead to the next state, it assumes it will take the *best* action there (the $\max$ below), regardless of what it actually ends up doing. Because of this, Q-Learning [provably](https://link.springer.com/article/10.1007/BF00992698) converges to the optimal action values $Q_*$, as long as the agent keeps trying every action in every state often enough (i.e., it keeps **exploring**).

    Within one episode, Q-Learning works as follows:

    1. Initialize $t = 0$
    2. Start in $S_0$
    3. At time step $t$, choose an action $A_t$ using a policy that mostly picks the action with the largest Q-value, but sometimes explores (e.g. the softmax policy below).
    4. After applying action $A_t$, we observe reward $R_{t+1}$ and encounter the next state $S_{t+1}$.
    5. Update the Q-value function according to:

    $$
    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max\limits_{a}Q(S_{t+1},a) - Q(S_t,A_t)]
    $$

    6. $t = t+1$ and repeat from Step 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional: attention-weighted RL

    *This is an extra example. Skip it if you are short on time.*

    So far, every state was a single, unique thing (a square on a track, a slot machine). Real stimuli have many **features**: a color, a shape, a texture. Which of them predict reward? Learning a separate value for every combination of features quickly becomes impossible. A simpler solution is to learn values for individual features, and to use **attention** to decide which features matter.

    [Leong, Radulescu et al. (2017)](https://www.cell.com/neuron/fulltext/S0896-6273(16)31039-X) studied this with the "dimensions task." On each trial, participants choose among three stimuli. Each stimulus has one feature on each of three dimensions (e.g., a face, a house and a tool). Only one feature on one dimension (the target) matters: choosing the stimulus that contains it is rewarded 75% of the time, and anything else is rewarded 25% of the time.

    In the **attention-weighted RL** model, the value of a stimulus $s$ is a weighted sum of the values $W$ of its features, one feature $f_d(s)$ per dimension $d$, with attention weights $\phi_d$ that sum to 1:

    $$
    V(s) = \sum_d \phi_d \, W\big(f_d(s)\big)
    $$

    After a choice, the prediction error $\delta = r - V(s_\text{chosen})$ updates each feature of the chosen stimulus, **in proportion to how much it was attended**:

    $$
    W\big(f_d\big) \leftarrow W\big(f_d\big) + \alpha \, \phi_d \, \delta
    $$

    In the paper, the attention weights $\phi$ were *measured* on each trial using eye-tracking and fMRI. To simulate the model here, we let attention follow value instead: the agent attends more to dimensions that contain a high-valued feature, with $\eta$ controlling how strongly. With $\eta = 0$, attention is spread evenly across dimensions (standard feature-based RL).

    $$
    \phi_d = \frac{\exp\big(\eta \max_f W_{d,f}\big)}{\sum_{d'} \exp\big(\eta \max_f W_{d',f}\big)}
    $$

    The agent chooses between the three stimuli with a softmax on $V$, with inverse temperature $\beta$ (more on softmax in the next section).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Discussion question*: Why does attention speed up learning in this task? Turn $\eta$ up to its maximum. Does strong attention always help, or can the agent get stuck attending to the wrong dimension?

    *Discussion question*: In this simulation, value drives attention and attention drives learning. Leong et al. found evidence for both directions in human participants. How could a bias in attention (e.g., toward threat-related features) change what someone learns?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From RL algorithms to modeling human behavior

    ### Simulating behavior

    To see how we can apply these ideas to modeling human behavioral data, let's consider a simple environment in which on each trial $t$, a participant makes a choice $c_t$ (= Left or Right) between a left and a right slot machine, and receives a reward $r_t$ (= \$1 or \$0) stochastically with probability $0.8$ if it chooses Left, or with probability $0.2$ if it chooses Right. This environment is called a **2-armed bandit**.

    Let's code up this environment below:
    """)
    return


@app.cell
def _(np, plt):
    class Bandit(object):
        """Class for the multiarmed bandit environment.

        Parameters
        ----------
        k : int
            Number of arms (actions), numbered 0..k-1.
        best : int
            Best action.
        p_best : float, range (0, 1)
            Reward probability for the best action.
        p_other : float, range (0, 1)
            Reward probability for the other arms.
        """

        def __init__(self, k, best, p_best, p_other, labels=None, rng=None):
            self.k = k  # number of arms
            self.best_action = best  # best action
            self.p_best = p_best  # probability of reward for best action
            self.p_other = p_other  # probability of reward for other actions
            self.t = 0  # initialize timestep
            self.labels = labels if labels is not None else [str(a) for a in range(k)]
            self.rng = rng if rng is not None else np.random.default_rng()

        def reward_probs(self):
            p_reward = np.ones((self.k,)) * self.p_other
            p_reward[self.best_action] = self.p_best
            return p_reward

        def visualize(self):
            # in the multiarmed bandit case, the structure of the environment
            # is the reward probability of each arm
            fig, ax = plt.subplots(figsize=(1.2 * self.k + 1, 3))
            ax.bar(np.arange(self.k), self.reward_probs(), color="#4c72b0")
            ax.set_xticks(np.arange(self.k))
            ax.set_xticklabels(self.labels)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Action")
            ax.set_ylabel("Reward probability")
            ax.spines[["top", "right"]].set_visible(False)
            return fig

        def step(self, action):
            # the step method takes as input an action and changes the environment
            # in the bandit case, the environment responds with a reward
            p = self.p_best if action == self.best_action else self.p_other
            reward = int(self.rng.random() < p)

            # increment timestep
            self.t = self.t + 1
            return reward

        def reset(self):
            # the reset method resets the environment to its start state
            self.t = 0

    return (Bandit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Exercise*: Use the bandit class to instantiate a 2-armed bandit in which the probability of reward for taking the best action is 0.8, and 0.2 otherwise. Visualize to verify.
    """)
    return


@app.cell
def _(Bandit):
    twoarmed = Bandit(2, 0, 0.8, 0.2, labels=["A", "B"])
    twoarmed.visualize()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's code up the Q-Learning agent. To choose actions, the agent uses a **softmax** policy, which turns Q-values into choice probabilities:

    $$
    P(a_t = a) = \frac{\exp(\beta \, Q(a))}{\sum_{a'} \exp(\beta \, Q(a'))}
    $$

    The **inverse temperature** $\beta \geq 0$ controls how deterministic choices are: with $\beta = 0$ the agent chooses randomly; as $\beta$ grows, it almost always picks the action with the highest Q-value. This is one simple way to balance **exploration** and **exploitation**.

    In the bandit there is only one state and each trial ends after one choice, so there is no next state to look ahead to ($\gamma$ drops out) and the Q-Learning update reduces to

    $$
    Q(a_t) \leftarrow Q(a_t) + \alpha\,[r_t - Q(a_t)]
    $$

    which is the classic Rescorla–Wagner (delta) rule.
    """)
    return


@app.cell
def _(np):
    class QLearning(object):
        """Q-learning agent for a k-armed bandit (a single state).

        Parameters
        ----------
        k : int
            Number of actions.
        alpha : float, range (0, 1)
            Learning rate.
        beta : float, range [0, inf)
            Inverse temperature of the softmax policy.
        """

        def __init__(self, k, alpha, beta, q_init=0.0, rng=None):
            self.k = k
            self.a = alpha
            self.b = beta
            self.q = np.ones((k,)) * q_init
            self.rng = rng if rng is not None else np.random.default_rng()

        def softmax(self, q_values):
            """Compute softmax probabilities for a vector of Q-values."""
            z = self.b * q_values
            z = z - np.max(z)  # numerical stability trick: subtract max
            exp_z = np.exp(z)
            return exp_z / np.sum(exp_z)

        def policy(self):
            """Select action according to softmax probabilities."""
            probs = self.softmax(self.q)
            return self.rng.choice(self.k, p=probs)

        def update(self, action, reward, verbose=False):
            delta = reward - self.q[action]
            self.q[action] = self.q[action] + self.a * delta
            if verbose:
                print(self.q)

    return (QLearning,)


@app.cell
def _(Bandit, QLearning, np):
    def simulate_bandit(alpha, beta, n_trials, p_best=0.8, p_other=0.2, seed=0):
        """Simulate a Q-learning agent on a 2-armed bandit (Left = 0 is best)."""
        rng = np.random.default_rng(seed)
        env = Bandit(2, 0, p_best, p_other, labels=["Left", "Right"], rng=rng)
        agent = QLearning(env.k, alpha, beta, rng=rng)

        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)
        Q = np.zeros((n_trials, env.k))  # Q-values *before* each choice

        for t in range(n_trials):
            Q[t] = agent.q
            choices[t] = agent.policy()
            rewards[t] = env.step(choices[t])
            agent.update(choices[t], rewards[t])

        return choices, rewards, Q

    return (simulate_bandit,)


@app.cell(hide_code=True)
def _(mo):
    bandit_alpha = mo.ui.slider(0.01, 1.0, step=0.01, value=0.3, label="learning rate α", show_value=True)
    bandit_beta = mo.ui.slider(0.0, 20.0, step=0.5, value=5.0, label="inverse temperature β", show_value=True)
    bandit_trials = mo.ui.slider(20, 500, step=10, value=200, label="trials", show_value=True)
    bandit_seed = mo.ui.number(0, 1000, step=1, value=0, label="random seed")
    mo.vstack([mo.md("**Simulated participant:**"), bandit_alpha, bandit_beta, bandit_trials, bandit_seed])
    return bandit_alpha, bandit_beta, bandit_seed, bandit_trials


@app.cell
def _(bandit_alpha, bandit_beta, bandit_seed, bandit_trials, simulate_bandit):
    choices, rewards, Q = simulate_bandit(
        bandit_alpha.value, bandit_beta.value, bandit_trials.value, seed=int(bandit_seed.value)
    )
    return Q, choices, rewards


@app.cell
def _(Q, choices, np, plt, rewards):
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _t = np.arange(len(choices))
    _ax.plot(_t, Q[:, 0], color="#4c72b0", label="Q(Left)  [p(reward) = 0.8]")
    _ax.plot(_t, Q[:, 1], color="#dd8452", label="Q(Right) [p(reward) = 0.2]")
    # choices as ticks: top row = Left, bottom row = Right; filled = rewarded
    for _c, _y, _col in [(0, 1.08, "#4c72b0"), (1, -0.08, "#dd8452")]:
        _m = choices == _c
        _ax.scatter(_t[_m & (rewards == 1)], np.full((_m & (rewards == 1)).sum(), _y), marker="|", color=_col, s=80)
        _ax.scatter(_t[_m & (rewards == 0)], np.full((_m & (rewards == 0)).sum(), _y), marker="|", color=_col, alpha=0.3, s=80)
    _ax.set_ylim(-0.15, 1.15)
    _ax.set_xlabel("Trial", fontsize=15)
    _ax.set_ylabel("Q-value", fontsize=15)
    _ax.set_title(f"Chose the better arm on {np.mean(choices == 0):.0%} of trials (ticks = choices; faded = unrewarded)")
    _ax.legend(loc="center right", frameon=False)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Discussion question*: What happens to behavior when $\beta$ is very low? Very high? What about when $\alpha$ is close to 1?

    ### The likelihood function

    In a typical experiment, our data consists of stimuli, participant choices, and outcomes. The likelihood function is the statistical model that allows us to relate the algorithm to the data we observe. In general, if the data generated by an RL algorithm matches the data we obtain from our participants, we say that the algorithm is a good description of the cognitive process.

    We need a mathematical way to specify how to link between variables of the algorithm and choice data. The most common *link function* is the softmax we already used to simulate choices [[1]](https://www.princeton.edu/~ndaw/d10.pdf)[[2]](https://elifesciences.org/articles/49547). To compute the likelihood, we replay the participant's *actual* choices and rewards through the model, and on every trial ask: what probability did the model assign to the choice the participant made?

    $$
    \log \mathcal{L}(\alpha, \beta) = \sum_t \log P(c_t \mid Q_t; \beta)
    $$

    The parameters that make the observed choices most probable are the **maximum likelihood estimates**.
    """)
    return


@app.cell
def _(np):
    def qlearning_choice_loglik(alpha, beta, choices, rewards, k=2, q_init=0.0, return_per_trial=False):
        """Log-likelihood of observed bandit choices under the Q-learning + softmax model.

        Parameters
        ----------
        alpha, beta : float
            Learning rate and softmax inverse temperature.
        choices : array of int, values 0..k-1
            Observed choices. Use -1 to mark a missed trial (no likelihood, no update).
        rewards : array of numeric
            Observed rewards.

        Returns
        -------
        total_loglik : float
        (optional) per_trial_logp : np.ndarray (np.nan for skipped trials)
        """
        choices = np.asarray(choices)
        rewards = np.asarray(rewards)
        assert len(choices) == len(rewards), "All inputs must be same length"

        q = np.ones(k) * q_init
        per_logp = np.full(len(choices), np.nan)

        for t, (c, r) in enumerate(zip(choices, rewards)):
            if c < 0:  # missed trial
                continue
            # numerically stable log-softmax
            z = beta * q
            z = z - np.max(z)
            per_logp[t] = z[c] - np.log(np.sum(np.exp(z)))
            # learn from the observed outcome
            q[c] = q[c] + alpha * (r - q[c])

        total_loglik = np.nansum(per_logp)
        return (total_loglik, per_logp) if return_per_trial else total_loglik

    return (qlearning_choice_loglik,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's evaluate the log-likelihood of the simulated choices above over a grid of $(\alpha, \beta)$ values. If the model is identifiable, the peak of this surface (white star) should land near the parameters we used to generate the data (black cross). Change the sliders above, or increase the number of trials, and see how the estimate changes.
    """)
    return


@app.cell
def _(
    bandit_alpha,
    bandit_beta,
    choices,
    np,
    plt,
    qlearning_choice_loglik,
    rewards,
):
    _alphas = np.linspace(0.02, 1.0, 40)
    _betas = np.linspace(0.0, 20.0, 41)
    _LL = np.array([[qlearning_choice_loglik(_a, _b, choices, rewards) for _a in _alphas] for _b in _betas])
    _i, _j = np.unravel_index(np.argmax(_LL), _LL.shape)

    _fig, _ax = plt.subplots(figsize=(6, 4.5))
    _im = _ax.pcolormesh(_alphas, _betas, _LL, shading="auto", cmap="viridis",
                         vmin=np.percentile(_LL, 30))
    _fig.colorbar(_im, ax=_ax, label="log-likelihood")
    _ax.scatter(bandit_alpha.value, bandit_beta.value, marker="x", color="black", s=100, label="true")
    _ax.scatter(_alphas[_j], _betas[_i], marker="*", color="white", edgecolor="black", s=200, label="best fit")
    _ax.set_xlabel("learning rate α", fontsize=13)
    _ax.set_ylabel("inverse temperature β", fontsize=13)
    _ax.set_title(f"best fit: α = {_alphas[_j]:.2f}, β = {_betas[_i]:.1f}")
    _ax.legend(loc="upper right")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Discussion question*: The likelihood surface often has a long diagonal ridge. What does that tell you about how well $\alpha$ and $\beta$ can be separately estimated from a single participant's data? How might you fix that in an experiment?

    In practice, instead of a grid, we would use an optimizer (e.g. `scipy.optimize.minimize` on the negative log-likelihood), and then check **parameter recovery**: simulate many participants with known parameters, fit them, and verify that we get back what we put in.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    ## Looking ahead: RL and the rest of cognition

    Another way to look at model-free and model-based RL:
    """),
        mo.image(src=mo.notebook_location() / "public" / "rl_three_time_frames.png", width=750),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project ideas

    **Your goal:** pick one of the two papers below, download its open data, and develop a model that **reproduces a key result**.

    ### Project 1: Attention and learning in ADHD and dyslexia

    **Paper:** Gabay, Jacob, Mansour & Hertz (2025). [Computational markers show specific deficits for dyslexia and ADHD in complex learning settings](https://www.nature.com/articles/s41539-025-00323-4). *npj Science of Learning*, 10, 38.

    **Data and code:** [osf.io/fq9um](https://osf.io/fq9um/) (CSV data and the authors' R model code).

    **Task:** Participants choose among three stimuli that vary in color, shape and texture. Within each game, one feature is rewarded 75% of the time and everything else 25% of the time. In the *No-Explicit-Assist* condition, participants must discover which dimension matters; in the *Explicit-Assist* condition, they are told.

    **Sample:** adults with ADHD (N = 21) and matched controls (N = 32), and adults with dyslexia (N = 20) and matched controls (N = 23).

    **Getting started:** the README on OSF explains the data format. The key columns are `reward`, `LCR` (the chosen stimulus: 1 = left, 2 = center, 3 = right), and columns 7–15, which give the shape, color and texture of the left, center and right stimuli.

    **Project - simulate models that reproduce the behavior and group differences in Figure 2:** the percentage of games learned in each condition (Explicit-Assist vs. No-Explicit-Assist), for each clinical group and its matched controls. A game counts as learned if the participant chose the stimulus with the target feature 8 times in a row; otherwise it ends after 25 trials. Chance is 33%. Check your numbers against the paper: about 80% of games learned with Explicit-Assist and 53% without. **Hint**: think about what modifications you would need to make to the Environment and Bandit classes in order to adapt it to the task in the paper.

    ### Project 2: Compulsivity and model-based control

    **Paper:** Gillan, Kosinski, Whelan, Phelps & Daw (2016). [Characterizing a psychiatric symptom dimension related to deficits in goal-directed control](https://elifesciences.org/articles/11305). *eLife*, 5, e11305.

    **Data:** [osf.io/usdgt](https://osf.io/usdgt/), also listed on [OpenCogData](https://nimh-dsst.github.io/OpenCogData/gillan-et-al-2016/).

    **Task:** the two-step task. A first-stage choice leads to one of two second-stage states, through a common (70%) or rare (30%) transition. Second-stage choices are rewarded with probabilities that drift slowly over time. A model-free learner simply repeats rewarded first-stage choices. A model-based learner takes the transition structure into account, so a reward after a rare transition makes it switch.

    **Sample:** two large online samples recruited on Amazon Mechanical Turk (Experiment 1: N = 548; Experiment 2: N = 1,413), each with the two-step task and self-report psychiatric questionnaires.

    **What they found:** model-based control was weaker in people with more self-reported compulsive symptoms. A factor analysis of 209 questionnaire items identified three symptom dimensions (*Anxious-Depression*, *Compulsive Behavior and Intrusive Thought*, and *Social Withdrawal*). Reduced model-based control was specifically linked to the compulsivity dimension.

    **Getting started:** each experiment folder has one task file per participant, the self-report scores, and a readme. In each task file, the task data start on the row after the one that says `twostep_instruct_9`. The columns include the trial number, the four drifting reward probabilities, the first-stage choice, whether the transition was common or rare, the second-stage state and choice, and the reward.

    **Project - simulate models that reproduce the behavior and group differences in Figure 4:** the probability of repeating the first-stage choice ("stay") as a function of whether the previous trial was rewarded and whether its transition was common or rare, for each experiment. A model-free learner stays after reward regardless of the transition; a model-based learner stays after a rewarded common transition but switches after a rewarded rare one. Participants show a mixture of both. **Hint**: think about how an agent would use the knowledge about transitions to compute values.

    ### Other examples

    [A model of mood as reward prediction errors](https://www.sciencedirect.com/science/article/pii/S1364661315001746)

    [Working-memory & RL interactionsin schizophrenia](https://pmc.ncbi.nlm.nih.gov/articles/PMC4188972/pdf/zns13747.pdf)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
