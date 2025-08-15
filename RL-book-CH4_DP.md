
# Reinforcement_Learning_class

### Chapter 4 Dynamic Programming

### **强化学习中的动态规划（Dynamic Programming, DP）**


## **1. 动态规划的核心思想**
在强化学习中，动态规划主要用于：
1. **预测（Prediction）**：计算给定策略 $( \pi )$ 的状态值函数 ( $v_\pi(s)$ ) 或动作值函数 ( $q_\pi(s, a)$ )。
2. **控制（Control）**：优化策略 (  $\pi$ )，找到最优策略 ( π <sup>*</sup> ) 和最优值函数 \( v_*(s) \) 或 \( q_*(s, a) \)。

动态规划的关键假设：
- **环境模型已知**（Model-Based）：即状态转移概率 \( p(s', r | s, a) \) 已知。
- **有限状态和动作空间**（适用于表格型方法）。

---

## **2. 动态规划的主要算法**

<img src="Ch4-r1.png" alt="State Transition Example" width="600"/> 
<img src="Ch4-r2.png" alt="State Transition Example" width="600"/> 
<img src="Ch4-r3.png" alt="State Transition Example" width="600"/> 

### **(1) 策略评估（Policy Evaluation）**
**输入 (Input)**: 给定策略 ( $\pi$ )

**目标**：计算给定策略 \( $\pi$ \) 的 state value \( $v_\pi(s)$ \)。  
**方法**：迭代应用贝尔曼方程：

$v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) \left[ r + \gamma v_k(s') \right]$
其中：
- \( v_k(s) \) 是第 \( k \) 次迭代时的值函数估计。
- \( $\gamma$ \) 是折扣因子。

### HW2

<img src="Ch4-r1.png" alt="State Transition Example" width="600"/> 

 
### **算法参数 (Algorithm parameter)**

* **Algorithm parameter: a small threshold $\theta > 0$ determining accuracy of estimation**
    * 一个小的正数 $\theta$。它是一个收敛阈值，用来决定算法何时停止。当价值函数的更新量小于这个阈值时，我们认为算法已经收敛，得到了一个足够精确的估算值。
### **初始化 (Initialize)**

* **Initialize $V(s)$ arbitrarily, for $s \in S$, and $V(terminal)$ to 0**
    * 对于每一个非终止状态 $s \in S$，将它的价值函数 $V(s)$ 初始化为一个任意值。通常可以初始化为0。
    * 对于终止状态 `terminal`，其价值函数 $V(terminal)$ 被固定为0。这是因为在终止状态下无法获得进一步的奖励，所以其价值为0。

### **主循环 (Loop)**

这是一个 `do-while` 或 `repeat-until` 类型的循环，它会反复执行直到满足退出条件。

* **Loop:**
    * **$\Delta \leftarrow 0$**：初始化一个变量 $\Delta$（Delta），用于记录这一次迭代中所有状态价值函数更新的最大变化量。每次迭代开始时都将其重置为0。
    * **Loop for each state $s \in S$:**：遍历所有的状态 $s$。
        * **$v \leftarrow V(s)$**：在更新 $V(s)$ 之前，将当前状态 $s$ 的旧价值函数值 $V(s)$ 存储到变量 $v$ 中。
        * **$V(s) \leftarrow \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a) [r + \gamma V(s')]$**
            * 这是贝尔曼期望方程 (Bellman Expectation Equation) 的更新形式。这是算法的核心。
            * **$\sum_a \pi(a|s)$**：对所有可能的动作 $a$ 求和，并用策略 $\pi(a|s)$ 作为权重。这反映了在状态 $s$ 下，策略 $\pi$ 选择不同动作的概率。
            * **$\sum_{s',r} p(s',r|s,a)$**：对所有可能的下一个状态 $s'$ 和即时奖励 $r$ 求和。这是环境的动态特性，`$p(s',r|s,a)$` 表示在状态 $s$ 采取动作 $a$ 后，转移到下一个状态 $s'$ 并获得奖励 $r$ 的概率。
            * **$[r + \gamma V(s')]$**：这是根据贝尔曼方程计算的期望回报。
                * $r$ 是即时奖励。
                * $\gamma$ 是折扣因子（gamma），一个介于0和1之间的数。它决定了未来奖励的重要性。
                * $V(s')$ 是下一个状态 $s'$ 的价值函数，这里用的是上一次迭代计算出的值。
        * **$\Delta \leftarrow \max(\Delta, |v - V(s)|)$**
            * 计算当前状态的价值函数更新量 $|v - V(s)|$。
            * 用这个更新量与当前的最大变化量 $\Delta$ 进行比较，并更新 $\Delta$ 为两者中的最大值。这样，在遍历完所有状态后，$\Delta$ 就记录了本次迭代中价值函数更新的最大幅度。

### **收敛条件 (until)**

* **until $\Delta < \theta$**
    * 当最大更新量 $\Delta$ 小于预设的阈值 $\theta$ 时，循环终止。这意味着价值函数的值已经非常稳定，接近于收敛。


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

<img src="Ch4-r2.png" alt="State Transition Example" width="600"/> 

### **(4) 值迭代（Value Iteration）**
**目标**：直接优化值函数，不显式计算策略，直到收敛到最优值函数 \( v_* \)。  
**方法**：迭代更新贝尔曼最优方程：
$$
v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) \left[ r + \gamma v_k(s') \right]
$$

<img src="Ch4-r3.png" alt="State Transition Example" width="600"/> 

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



这是一个名为“**迭代策略评估 (Iterative Policy Evaluation)**”的算法伪代码，用于估算给定策略 $\pi$ 的价值函数 $V \approx V_\pi$。

下面我来详细解释一下它的每个部分：

### **算法名称**

* **Iterative Policy Evaluation, for estimating $V \approx V_\pi$**
    * **迭代策略评估**：这是一个重复执行直到收敛的算法。
    * **用于估算 $V \approx V_\pi$**：它的目标是计算一个策略 $\pi$ 的价值函数 $V_\pi$。价值函数 $V_\pi(s)$ 表示从状态 $s$ 开始，遵循策略 $\pi$ 所能获得的长期累积回报（奖励）的期望值。

### **输入 (Input)**

* **Input $\pi$, the policy to be evaluated**
    * 输入是一个策略 $\pi$。策略 $\pi(a|s)$ 定义了在给定状态 $s$ 下，选择动作 $a$ 的概率。这是我们想要评估的策略。





### **总结**

这个算法本质上是**动态规划 (Dynamic Programming)** 的一种应用。它通过反复应用贝尔曼期望方程来更新每个状态的价值函数。每一轮迭代都会利用当前最好的价值函数估计值来计算一个新的、更精确的估计值。通过不断地迭代，价值函数的值会逐渐收敛到真实的 $V_\pi$。当更新量变得非常小，小于设定的阈值 $\theta$ 时，算法停止，我们得到的 $V(s)$ 就被认为是策略 $\pi$ 的价值函数 $V_\pi(s)$ 的一个良好估计。