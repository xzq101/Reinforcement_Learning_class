def n_step_td(V_initial, t, n, rewards, gamma=1.0, alpha=0.1, terminal_value=0):
    """
    计算 n-step TD 的状态价值更新（支持向量输入）
    
    参数:
        V_initial (list): 各状态的初始价值向量 [V(S_0), V(S_1), ..., V(S_{T-1})]
        t (int): 要更新的时间步（从 0 开始）
        n (int): n-step 的 n
        rewards (list): 奖励序列 [R_1, R_2, ..., R_T]
        gamma (float): 折扣因子，默认为 1.0
        alpha (float): 学习率，默认为 0.1
        terminal_value (float): 终止状态的价值，默认为 0
    
    返回:
        float: 更新后的状态价值 V(S_t)
    """
    T = len(rewards)  # episode 的总时间步
    
    # 检查输入合法性
    if len(V_initial) != T:
        raise ValueError("V_initial 的长度必须与 rewards 的长度相同！")
    if t < 0 or t >= T:
        raise ValueError("时间步 t 超出范围！")
    
    # 计算 G_{t:t+n}
    G = 0.0
    for k in range(1, n + 1):
        current_step = t + k
        
        # 如果超出 episode 长度，则终止
        if current_step > T:
            break
        
        # 累加 gamma^{k-1} * R_{t+k}
        G += (gamma ** (k - 1)) * rewards[current_step - 1]
    
    # 如果 t + n 未超出 episode 长度，则加上 gamma^n * V(S_{t+n})
    if t + n < T:
        G += (gamma ** n) * V_initial[t + n]
    else:
        # 如果 t + n 超出 episode 长度，则 V(终止) = terminal_value
        G += (gamma ** (T - t)) * terminal_value
    
    # 更新 V(S_t)
    V_updated = V_initial[t] + alpha * (G - V_initial[t])
    
    return V_updated


def update_all_states(V_initial, n, rewards, gamma=1.0, alpha=0.1):
    """
    更新所有状态的价值（向量输入）
    """
    T = len(rewards)
    V = V_initial.copy()  # 避免修改原数组
    
    for t in range(T):
        V[t] = n_step_td(V, t, n, rewards, gamma, alpha)
    
    return V


# 示例使用
if __name__ == "__main__":
    # 给定参数
    V_initial = [30, 25, 20, 15, 10]  # 初始状态价值向量
    print("初始状态价值向量:", V_initial)
    n = 3                                  # n-step 的 n
    rewards = [0, 0, 0, 0, 1]              # 奖励序列 [R_1, R_2, ..., R_5]
    gamma = 0.9                             # 折扣因子
    alpha = 0.1                             # 学习率

    # 更新所有状态
    V_final = update_all_states(V_initial, n, rewards, gamma=gamma, alpha=alpha)
    print("更新后的状态价值向量:", V_final)