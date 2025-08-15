
# Reinforcement_Learning_class

### Chapter 4 Dynamic Programming

### **强化学习中的动态规划（Dynamic Programming, DP）**


## **1. 动态规划的核心思想**
在强化学习中，动态规划主要用于：
1. **预测（Prediction）**：计算给定策略 $( \pi )$ 的状态值函数 ( $v_\pi(s)$ ) 或动作值函数 \( q_\pi(s, a) \)。
2. **控制（Control）**：优化策略 \( \pi \)，找到最优策略 \( \pi^* \) 和最优值函数 \( v_*(s) \) 或 \( q_*(s, a) \)。

动态规划的关键假设：
- **环境模型已知**（Model-Based）：即状态转移概率 \( p(s', r | s, a) \) 已知。
- **有限状态和动作空间**（适用于表格型方法）。

---

## **2. 动态规划的主要算法**
### **(1) 策略评估（Policy Evaluation）**
**目标**：计算给定策略 \( \pi \) 的状态值函数 \( v_\pi(s) \)。  
**方法**：迭代应用贝尔曼方程：
\[
v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) \left[ r + \gamma v_k(s') \right]
\]
其中：
- \( v_k(s) \) 是第 \( k \) 次迭代时的值函数估计。
- \( \gamma \) 是折扣因子。

**伪代码**：
```python
def policy_evaluation(pi, p, gamma=0.9, theta=1e-6):
    v = np.zeros(n_states)  # 初始化值函数
    while True:
        delta = 0
        for s in states:
            old_v = v[s]
            v[s] = sum(pi(a|s) * sum(p(s',r|s,a) * (r + gamma * v[s']) for s',r in transitions) for a in actions)
            delta = max(delta, abs(v[s] - old_v))
        if delta < theta:
            break
    return v
```

---

### **(2) 策略改进（Policy Improvement）**
**目标**：基于当前值函数 \( v_\pi(s) \)，改进策略 \( \pi \)：
\[
\pi'(s) = \arg\max_a \sum_{s', r} p(s', r | s, a) \left[ r + \gamma v_\pi(s') \right]
\]
即，在每个状态选择**贪心动作**（使未来回报最大化的动作）。

---

### **(3) 策略迭代（Policy Iteration）**
**目标**：交替进行**策略评估**和**策略改进**，直到策略收敛到最优 \( \pi^* \)。

**步骤**：
1. **初始化**：随机策略 \( \pi_0 \)。
2. **策略评估**：计算 \( v_{\pi_k} \)。
3. **策略改进**：生成新策略 \( \pi_{k+1} \)（贪心策略）。
4. **重复**，直到策略不再变化。

**伪代码**：
```python
def policy_iteration(p, gamma=0.9):
    pi = np.random.choice(actions, size=n_states)  # 随机初始化策略
    while True:
        v = policy_evaluation(pi, p, gamma)  # 策略评估
        pi_new = {}
        for s in states:
            # 策略改进：选择最优动作
            pi_new[s] = argmax_a(sum(p(s',r|s,a) * (r + gamma * v[s']) for s',r in transitions))
        if pi_new == pi:
            break
        pi = pi_new
    return pi, v
```

---

### **(4) 值迭代（Value Iteration）**
**目标**：直接优化值函数，不显式计算策略，直到收敛到最优值函数 \( v_* \)。  
**方法**：迭代更新贝尔曼最优方程：
\[
v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) \left[ r + \gamma v_k(s') \right]
\]
**伪代码**：
```python
def value_iteration(p, gamma=0.9, theta=1e-6):
    v = np.zeros(n_states)
    while True:
        delta = 0
        for s in states:
            old_v = v[s]
            v[s] = max(sum(p(s',r|s,a) * (r + gamma * v[s']) for s',r in transitions) for a in actions)
            delta = max(delta, abs(v[s] - old_v))
        if delta < theta:
            break
    # 提取最优策略
    pi = {}
    for s in states:
        pi[s] = argmax_a(sum(p(s',r|s,a) * (r + gamma * v[s']) for s',r in transitions))
    return pi, v
```

---

## **3. 动态规划的优缺点**
### **优点**
✅ **数学保证**：在已知 MDP 模型时，能收敛到最优策略。  
✅ **高效**：相比蒙特卡洛（MC）或时序差分（TD），DP 计算更快（适用于小规模问题）。  

### **缺点**
❌ **需要完整的环境模型**（即 \( p(s', r | s, a) \) 已知）。  
❌ **计算复杂度高**（状态空间大时，计算量爆炸）。  
❌ **不适合连续状态/动作空间**（需离散化或函数逼近）。

---

## **4. DP 在强化学习中的应用**
- **经典控制问题**：如 Grid World、Frozen Lake、CartPole（离散化后可用 DP）。  
- **算法改进**：许多 RL 算法（如 Q-Learning、SARSA）受 DP 启发。  
- **理论研究**：DP 是理解贝尔曼方程和值函数迭代的基础。

---

## **总结**
| 方法 | 适用场景 | 特点 |
|------|---------|------|
| **策略评估** | 计算给定策略的值函数 | 迭代贝尔曼方程 |
| **策略迭代** | 优化策略（策略评估 + 策略改进） | 收敛较慢但稳定 |
| **值迭代** | 直接优化值函数 | 更快收敛，但需计算最大值 |

动态规划是强化学习的**理论基础**，尽管实际应用受限（需已知模型），但它启发了许多现代 RL 算法（如 DQN、Policy Gradient）。