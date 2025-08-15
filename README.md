# Reinforcement_Learning_class

## Value Function Derivation in Markov Decision Process (MDP)

### 3.5 Bellman equation for v<sub>π</sub>.


v<sub>π</sub>(s) = Eπ[G_t | S_t = s]
= Eπ[R_{t+1} + γ G_{t+1} | S_t = s]
= Σₐ π(a|s) Σ_{s', r} p(s', r | s, a) [r + γ Eπ[G_{t+1} | S_{t+1} = s']]
= Σₐ π(a|s) Σ_{s', r} p(s', r | s, a) [r + γ vπ(s')], for all s ∈ S


The value function \( v_{\pi}(s) \) under policy \( \pi \) is defined and derived as follows:

\[
\begin{aligned}
v_{\pi}(s) &= \mathbb{E}_{\pi}[G_t \mid S_t = s] \\
&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
&= \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r|s, a) \left[ r + \gamma \mathbb{E}_{\pi}[G_{t+1}|S_{t+1} = s'] \right] \\
&= \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a) \left[ r + \gamma v_{\pi}(s') \right], \quad \text{for all } s \in \mathbb{S},
\end{aligned}
\]

This derivation corresponds to equations (3.9) and (3.14) from the referenced text.
