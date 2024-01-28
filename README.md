# 遗传算法求解背包问题

这是一个使用Python的DEAP库实现的遗传算法，用于解决经典的背包问题。背包问题的目标是在给定的一组物品中，选择一些物品以使它们的总重量不超过最大允许重量，同时最大化这些物品的总价值。

### 问题描述

- 物品重量：[10, 20, 30, 40, 50]
- 物品价值：[60, 100, 120, 200, 300]
- 最大允许重量：100

### 遗传算法步骤

1. **初始化种群**：创建了一个包含50个个体的初始种群，每个个体表示一个物品的选择方案。

2. **适应度函数**：定义了一个适应度函数 `evaluate`，用于计算每个个体的总重量和总价值，并根据最大允许重量来评估适应度。

3. **选择**：使用锦标赛选择方法来选择个体，根据其适应度值选择下一代的个体。

4. **交叉**：对选定的个体进行两点交叉，以生成下一代的个体。

5. **变异**：对选定的个体进行位翻转变异，以增加种群的多样性。

6. **进化迭代**：重复执行选择、交叉和变异步骤，进化种群100代。

7. **最优解**：从最终种群中选择最佳个体，输出最佳个体的选择方案和总价值。

### 结果

运行遗传算法后，找到了最优解：

- 最佳个体: [1, 1, 1, 0, 0]
- 最大总价值: 280

这意味着在最大允许重量为100的情况下，最佳选择是选择前三个物品，总重量为60，总价值为280。

这段代码演示了如何使用遗传算法来解决组合优化问题，特别是背包问题。通过不断进化种群，算法能够找到最优解，以满足约束条件并最大化目标函数。
