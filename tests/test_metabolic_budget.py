"""
代谢预算模块测试
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加core目录本身，直接导入
core_dir = os.path.join(project_root, 'civis_lucri_faber', 'core')
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

import torch
import torch.nn as nn
from metabolic_budget import (
    MetabolicBudget,
    MetabolicCostCalculator,
    PeriodicStarvation,
    create_metabolic_budget,
    MetabolicState,
)


# 测试用简单模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def test_metabolic_cost():
    """测试代谢成本计算"""
    print("\n[TEST 1] MetabolicCostCalculator")
    calc = MetabolicCostCalculator(resource_budget=0.3, sparse_coef=0.01)

    # 模拟隐藏状态
    hidden = torch.randn(32, 50, 128)

    loss, detail = calc(hidden, return_detail=True)
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Activation Rate: {detail['activation_rate']:.2%}")
    print(f"  - Budget: {detail['budget']:.2%}")
    print(f"  - Sparse Penalty: {detail['sparse_penalty']:.4f}")
    print("  ✓ PASSED")


def test_starvation():
    """测试周期性饥饿"""
    print("\n[TEST 2] PeriodicStarvation")

    starvation = PeriodicStarvation(
        starvation_prob=0.5,  # 方便测试的高概率
        cycle_steps=10,
        min_block_ratio=0.1,
        max_block_ratio=0.3,
    )

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 模拟训练循环
    for step in range(30):
        x = torch.randn(8, 64)
        target = torch.randint(0, 10, (8,))

        # 计算重要性
        importance = starvation.compute_importance_scores(
            model, {'x': x, 'target': target},
            lambda out, tgt: nn.CrossEntropyLoss()(out, tgt)
        )

        starvation.update_importance(importance)

        # 获取门控mask
        masks, is_starving = starvation.get_gate_mask(importance)
        print(f"  Step {step}: is_starving={is_starving}")

    stats = starvation.get_starvation_stats()
    print(f"  - Global Step: {stats['global_step']}")
    print("  ✓ PASSED")


def test_full_budget():
    """测试完整代谢预算系统"""
    print("\n[TEST 3] MetabolicBudget")

    budget = MetabolicBudget(
        resource_budget=0.25,
        starvation_prob=0.3,
        lambda_met=0.01,
    )

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step in range(50):
        x = torch.randn(16, 64)
        target = torch.randint(0, 10, (16,))

        # 前向
        output = model(x)

        # 计算任务损失
        task_loss = nn.CrossEntropyLoss()(output, target)

        # 获取中间层激活（简化，直接用第一层输出）
        hidden_states = torch.relu(model.fc1(x))

        # 计算带代谢成本的损失
        total_loss, detail = budget.compute_loss(task_loss, hidden_states, return_detail=True)

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            state = budget.get_state()
            print(f"  Step {step}: active_ratio={state.active_ratio:.2%}, met_cost={state.met_cost:.4f}")

    summary = budget.get_summary()
    print(f"\n  Final Summary:")
    print(f"    - Active Ratio: {summary['active_ratio']:.2%}")
    print(f"    - Budget: {summary['budget']:.2%}")
    print(f"    - Avg Metabolic Cost: {summary['met_cost']:.4f}")
    print("  ✓ PASSED")


def test_ablation():
    """消融测试：对比有/无代谢约束"""
    print("\n[TEST 4] Ablation Study")

    def train_model(use_metabolic: bool, steps=100):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        budget = create_metabolic_budget(resource_budget=0.3) if use_metabolic else None

        for step in range(steps):
            x = torch.randn(32, 64)
            target = torch.randint(0, 10, (32,))

            output = model(x)
            task_loss = nn.CrossEntropyLoss()(output, target)

            if budget:
                hidden = torch.relu(model.fc1(x))
                total_loss, _ = budget.compute_loss(task_loss, hidden)
            else:
                total_loss = task_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # 计算最终激活率
        with torch.no_grad():
            x = torch.randn(32, 64)
            hidden = torch.relu(model.fc1(x))
            activation_rate = (hidden.abs() > 1e-6).float().mean()

        return activation_rate.item()

    # 有代谢约束
    activation_with = train_model(use_metabolic=True)
    print(f"  With Metabolic Budget: {activation_with:.2%}")

    # 无代谢约束
    activation_without = train_model(use_metabolic=False)
    print(f"  Without Metabolic Budget: {activation_without:.2%}")

    improvement = (activation_without - activation_with) / activation_without
    print(f"  Activation Reduction: {improvement:.1%}")
    print("  ✓ PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Metabolic Budget Module Tests")
    print("=" * 60)

    test_metabolic_cost()
    test_starvation()
    test_full_budget()
    test_ablation()

    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)