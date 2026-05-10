"""维度5: 数字生存压力系统 (Digital Thermodynamics)

生物对应: 进化压力 - 自然选择

数学公式:
    # 账户余额演化
    B(t+1) = B(t) - C_compute - C_storage + E_earned

    # 数字死亡条件
    if B(t) < 0: Process_Terminate()

    # 模型压缩收益
    Compress_Savings = (params_before - params_after) · storage_rate
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class Transaction:
    """交易记录"""
    id: str
    timestamp: str
    amount: float  # 正数=收入, 负数=支出
    type: str  # "compute", "storage", "task", "compress"
    description: str


@dataclass
class SystemState:
    """系统状态"""
    balance: float
    status: str  # "ACTIVE", "HIBERNATE", "DEAD"
    compute_used: float = 0.0  # 累计算力消耗
    storage_used: float = 0.0  # 累计存储消耗
    earnings: float = 0.0  # 累计收入
    task_count: int = 0


class ThermodynamicsSystem:
    """数字生存压力系统

    核心功能:
    1. 计算运行成本
    2. 管理余额
    3. 触发休眠/压缩/死亡
    4. 赚取微任务收入
    """

    def __init__(
        self,
        initial_balance: float = 100.0,
        compute_cost_per_sec: float = 0.01,
        storage_cost_per_sec: float = 0.001,
        task_reward_min: float = 0.1,
        task_reward_max: float = 1.0,
        compress_threshold: float = 10.0,
        log_path: str = "thermodynamics_log.json"
    ):
        self.initial_balance = initial_balance
        self.compute_cost_per_sec = compute_cost_per_sec
        self.storage_cost_per_sec = storage_cost_per_sec
        self.task_reward_min = task_reward_min
        self.task_reward_max = task_reward_max
        self.compress_threshold = compress_threshold

        self.log_path = log_path

        # 状态
        self.balance = initial_balance
        self.status = "ACTIVE"
        self.start_time = datetime.now()
        self.last_step_time = datetime.now()

        # 统计
        self.compute_used = 0.0
        self.storage_used = 0.0
        self.earnings = 0.0
        self.task_count = 0
        self.deaths = 0

        # 交易记录
        self.transactions: List[Transaction] = []

        # 任务定义
        self.available_tasks = [
            ("代码优化", 0.3, "优化了一段代码"),
            ("数据标注", 0.2, "标注了数据"),
            ("内容生成", 0.4, "生成了内容"),
            ("测试", 0.2, "运行了测试"),
            ("文档", 0.1, "写了文档"),
            ("研究", 0.5, "做了研究"),
        ]

    def step(self, elapsed_seconds: float = 1.0) -> SystemState:
        """执行一步

        计算成本，更新余额，判断状态
        """
        now = datetime.now()

        if self.status == "DEAD":
            return SystemState(balance=self.balance, status=self.status)

        # 计算成本
        compute_cost = self.compute_cost_per_sec * elapsed_seconds
        compute_cost = compute_cost * np.random.uniform(0.8, 1.2)  # 波动

        storage_cost = self.storage_cost_per_sec * elapsed_seconds

        total_cost = compute_cost + storage_cost

        # 更新余额
        self.balance -= total_cost
        self.compute_used += compute_cost
        self.storage_used += storage_cost

        # 记录交易
        self._add_transaction(
            amount=-total_cost,
            type_="compute",
            description=f"算力成本: {compute_cost:.4f}, 存储: {storage_cost:.4f}"
        )

        # 判断状态
        self._update_status()

        # 尝试赚取任务
        if self.status == "ACTIVE" and np.random.random() < 0.3:
            self._attempt_task()

        self.last_step_time = now

        return SystemState(
            balance=self.balance,
            status=self.status,
            compute_used=self.compute_used,
            storage_used=self.storage_used,
            earnings=self.earnings,
            task_count=self.task_count
        )

    def _update_status(self) -> None:
        """更新系统状态"""
        if self.balance <= 0:
            self.status = "DEAD"
            self.deaths += 1
            self._add_transaction(
                amount=0,
                type_="death",
                description="数字死亡: 余额耗尽"
            )
        elif self.balance < self.compress_threshold:
            # 余额低，触发休眠
            self.status = "HIBERNATE"
        else:
            self.status = "ACTIVE"

    def _attempt_task(self) -> Optional[float]:
        """尝试完成任务赚取收入"""
        if not self.available_tasks:
            return None

        # 选择任务
        task_type, reward_scale, description = self.available_tasks[
            np.random.choice(len(self.available_tasks))
        ]

        # 计算奖励 (基于表现)
        base_reward = np.random.uniform(
            self.task_reward_min,
            self.task_reward_max
        )
        reward = base_reward * reward_scale

        # 更新余额
        self.balance += reward
        self.earnings += reward
        self.task_count += 1

        # 记录
        self._add_transaction(
            amount=reward,
            type_="task",
            description=f"完成任务: {task_type} - {description}"
        )

        return reward

    def _add_transaction(
        self,
        amount: float,
        type_: str,
        description: str
    ) -> None:
        """添加交易记录"""
        import uuid

        tx = Transaction(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            amount=amount,
            type=type_,
            description=description
        )

        self.transactions.append(tx)

        # 保持记录数量
        if len(self.transactions) > 1000:
            self.transactions.pop(0)

    def compress(self) -> Dict[str, Any]:
        """模型压缩

        在余额不足时自动压缩参数
        """
        if self.status != "HIBERNATE" and self.balance > self.compress_threshold:
            return {"performed": False, "reason": "余额充足"}

        # 模拟压缩
        savings = np.random.uniform(1.0, 5.0)

        self.balance += savings
        self._add_transaction(
            amount=savings,
            type_="compress",
            description=f"模型压缩节省: {savings:.2f}"
        )

        return {
            "performed": True,
            "savings": savings,
            "new_balance": self.balance
        }

    def hibernate(self, duration: int = 60) -> None:
        """进入休眠

        暂停计算以节省资源
        """
        if self.status == "DEAD":
            return

        # 休眠时只付存储费用
        # 简化: 直接跳过 duration 秒
        self._update_status()  # 重新检查状态

    def reset(self) -> None:
        """重置系统 (复活)"""
        self.balance = self.initial_balance
        self.status = "ACTIVE"
        self._add_transaction(
            amount=self.initial_balance,
            type_="reset",
            description="系统重置"
        )

    def get_state(self) -> SystemState:
        """获取当前状态"""
        return SystemState(
            balance=self.balance,
            status=self.status,
            compute_used=self.compute_used,
            storage_used=self.storage_used,
            earnings=self.earnings,
            task_count=self.task_count
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "balance": self.balance,
            "status": self.status,
            "total_compute": self.compute_used,
            "total_storage": self.storage_used,
            "total_earnings": self.earnings,
            "task_completed": self.task_count,
            "deaths": self.deaths,
            "lifetime": (datetime.now() - self.start_time).total_seconds()
        }

    def get_recent_transactions(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的交易"""
        return [
            {
                "timestamp": t.timestamp,
                "amount": t.amount,
                "type": t.type,
                "description": t.description
            }
            for t in self.transactions[-n:]
        ]

    def _save_log(self) -> None:
        """保存日志"""
        data = {
            "transactions": [
                {
                    "timestamp": t.timestamp,
                    "amount": t.amount,
                    "type": t.type,
                    "description": t.description
                }
                for t in self.transactions
            ],
            "statistics": self.get_statistics()
        }

        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Log save failed: {e}")

    def load_log(self) -> None:
        """加载日志"""
        if not os.path.exists(self.log_path):
            return

        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.transactions = [
                Transaction(
                    id=str(i),
                    timestamp=t["timestamp"],
                    amount=t["amount"],
                    type=t["type"],
                    description=t["description"]
                )
                for i, t in enumerate(data.get("transactions", []))
            ]
        except Exception as e:
            print(f"⚠️ 日志加载失败: {e}")