"""
小脑-脊髓运动控制系统 (Cerebellum-Spinal Cord)

对应生物学的运动控制层级：
1. 脊髓 - 反射弧、CPG(中央模式发生器)
2. 小脑 - 运动学习、时序整合、感觉运动映射
3. 协同 - 小脑→脊髓学习下行 + 脊髓→小脑反馈

核心功能：
1. 脊髓反射通路
2. 中央模式生成器(CPG)
3. 小脑运动学习
4. 感觉运动整合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from collections import deque


# ============ 脊髓系统 (Spinal Cord) ============

@dataclass
class MotorCommand:
    """运动指令"""
    joint_angles: np.ndarray       # 关节角度
    joint_velocities: np.ndarray  # 关节速度
    muscle_activations: np.ndarray  # 肌肉激活
    timestamp: int


@dataclass
class ReflexResponse:
    """反射响应"""
    reflex_type: str       # "stretch", "withdrawal", "crossed_extensor"
    amplitude: float
    latency: float        # 反射潜伏期(秒)
    duration: float


class CentralPatternGenerator(nn.Module):
    """
    中央模式发生器 (CPG)

    脊髓固有振荡器，产生节律性运动模式：
    - 行走、跑步、游泳等节律运动
    - 无需意识控制
    - 通过神经元振荡网络实现
    """

    def __init__(
        self,
        n_joints: int = 4,
        osc_frequency: float = 2.0,  # Hz, 约2Hz对应行走
        coupling_strength: float = 0.5,
    ):
        super().__init__()

        self.n_joints = n_joints
        self.frequency = osc_frequency
        self.coupling = coupling_strength

        # 每个关节一个振荡器
        self.phases = nn.Parameter(torch.zeros(n_joints))
        self.amplitudes = nn.Parameter(torch.ones(n_joints) * 0.5)

        # 耦合矩阵 (相邻关节相互抑制)
        coupling_matrix = torch.eye(n_joints)
        for i in range(n_joints - 1):
            coupling_matrix[i, i+1] = coupling_strength
            coupling_matrix[i+1, i] = coupling_strength
        self.coupling_matrix = coupling_matrix

        # 状态
        self.time = 0.0
        self.is_active = False

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成运动模式

        Returns:
            positions: 关节位置
            velocities: 关节速度
        """
        if not self.is_active:
            return torch.zeros(self.n_joints), torch.zeros(self.n_joints)

        # 2πf * t + phase
        phases = self.phases + self.time * self.frequency * 2 * np.pi

        # 正弦波输出
        positions = self.amplitudes * torch.sin(phases)
        velocities = self.amplitudes * self.frequency * 2 * np.pi * torch.cos(phases)

        self.time += 0.01  # 10ms步长

        return positions, velocities

    def set_phase_offset(self, joint_idx: int, offset: float):
        """设置相位偏移 (制造步态)"""
        self.phases.data[joint_idx] = offset

    def set_amplitude(self, joint_idx: int, amp: float):
        """设置振幅"""
        self.amplitudes.data[joint_idx] = amp

    def activate(self):
        """激活CPG"""
        self.is_active = True

    def deactivate(self):
        """停用"""
        self.is_active = False


class ReflexPathway(nn.Module):
    """
    脊髓反射通路

    1. 肌梭反射 (Stretch Reflex) - 快速调节
    2. 屈肌反射 (Withdrawal Reflex) - 躲避��害
    3. 交叉伸肌反射 (Crossed Extensor) - 姿势维持
    """

    def __init__(
        self,
        n_joints: int = 4,
        reflex_gain: float = 1.0,
    ):
        super().__init__()

        self.n_joints = n_joints
        self.reflex_gain = reflex_gain

        # 反射阈值
        self.stretch_threshold = 0.1  # 肌梭阈值
        self.withdrawal_threshold = 0.8  # 疼痛阈值

        # 延迟 (模拟突触延迟)
        self.latency = 0.03  # 30ms

    def compute_stretch_reflex(
        self,
        muscle_length: np.ndarray,
        muscle_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        肌梭反射

        肌肉被拉长时自动收缩抵抗
        对应位置伺服机制
        """
        # 长度变化 → 收缩抵抗
        stretch_response = -muscle_velocity * self.reflex_gain

        # 位置反馈
        stretch_response -= (muscle_length - 0.5) * 0.5

        return stretch_response

    def compute_withdrawal_reflex(
        self,
        pain_signal: float,
    ) -> np.ndarray:
        """
        屈肌反射

        疼痛 → 收缩屈肌、放松伸肌
        """
        if pain_signal < self.withdrawal_threshold:
            return np.zeros(self.n_joints)

        # 躲避方向 (简化)
        reflex = np.zeros(self.n_joints)
        reflex[0] = pain_signal * self.reflex_gain  # 收缩

        return reflex

    def compute_reflex(
        self,
        joint_angles: np.ndarray,
        joint_velocities: np.ndarray,
        sensory_input: np.ndarray = None,
    ) -> np.ndarray:
        """
        计算综合反射输出

        Returns:
            reflex_output: 反射调整量
        """
        reflex = np.zeros(self.n_joints)

        # 1. 肌梭反射 (每个关节)
        for i in range(self.n_joints):
            length = joint_angles[i]
            velocity = joint_velocities[i]
            reflex[i] += self.compute_stretch_reflex(
                np.array([length]),
                np.array([velocity])
            )[0]

        # 2. 伤害感受反射
        if sensory_input is not None:
            pain = np.mean(sensory_input)
            reflex += self.compute_withdrawal_reflex(pain)

        return np.clip(reflex, -1.0, 1.0)


class SpinalCord(nn.Module):
    """
    完整脊髓系统

    整合CPG + 反射通路 + 运动神经元
    """

    def __init__(
        self,
        n_joints: int = 4,
        cpg_frequency: float = 2.0,
    ):
        super().__init__()

        self.n_joints = n_joints

        # 中央模式生成器
        self.cpg = CentralPatternGenerator(
            n_joints=n_joints,
            osc_frequency=cpg_frequency,
        )

        # 反射通路
        self.reflex = ReflexPathway(n_joints=n_joints)

        # 运动神经元 (最终输出层)
        self.motor_neurons = nn.Sequential(
            nn.Linear(n_joints, n_joints),
            nn.Tanh()
        )

        # 状态
        self.joint_positions = np.zeros(n_joints)
        self.joint_velocities = np.zeros(n_joints)

    def forward(
        self,
        cpg_command: bool = False,
        reflex_input: np.ndarray = None,
    ) -> MotorCommand:
        """
        执行运动输出

        Args:
            cpg_command: 是否启用CPG
            reflex_input: 感觉输入

        Returns:
            motor_cmd: 运动指令
        """
        # 1. CPG产生
        if cpg_command:
            self.cpg.activate()
            cpg_pos, cpg_vel = self.cpg.forward()
            cpg_output = cpg_pos
        else:
            self.cpg.deactivate()
            cpg_output = torch.zeros(self.n_joints)

        # 2. 反射调整
        reflex_output = self.reflex.compute_reflex(
            self.joint_positions,
            self.joint_velocities,
            reflex_input,
        )

        # 3. 综合运动命令
        motor_output = cpg_output.float() + torch.tensor(reflex_output, dtype=torch.float32)
        motor_output = self.motor_neurons(motor_output)
        motor_output_detached = motor_output.detach()

        # 4. 更新状态
        self.joint_velocities = motor_output_detached.numpy() * 0.1
        self.joint_positions += self.joint_velocities * 0.01

        return MotorCommand(
            joint_angles=self.joint_positions.copy(),
            joint_velocities=self.joint_velocities.copy(),
            muscle_activations=motor_output_detached.numpy(),
            timestamp=0,
        )

    def update_sensors(
        self,
        joint_angles: np.ndarray,
        joint_velocities: np.ndarray,
    ):
        """更新本体感觉"""
        self.joint_positions = joint_angles
        self.joint_velocities = joint_velocities

    def get_proprioception(self) -> Dict:
        """获取本体感觉"""
        return {
            'joint_angles': self.joint_positions,
            'joint_velocities': self.joint_velocities,
        }


# ============ 小脑系统 (Cerebellum) ============

@dataclass
class ProceduralSkill:
    """程序性技能 (小脑的肌肉记忆)"""
    context_hash: int           # 上下文哈希
    action: int                 # 关联动作
    motor_pattern: np.ndarray   # 运动模式
    skill_level: float          # 熟练度 0-1
    auto_execution_count: int = 0  # 自动执行次数


class ProceduralMemory:
    """
    程序性记忆系统 (小脑肌肉记忆)

    对应"熟能生巧"的后半段：
    BG将重复达标的技能转移到这里，小脑接管自动执行。

    生物对应：
    - 骑自行车学会后不需要思考平衡 → 小脑自动控制
    - 打字熟练后不需要看键盘 → 小脑接管手指运动
    - 释放前额叶/BG的意识资源给新任务
    """

    def __init__(self, max_skills: int = 50):
        self.skills: List[ProceduralSkill] = []
        self.max_skills = max_skills

    def store(
        self,
        context: np.ndarray,
        action: int,
        motor_pattern: np.ndarray,
        skill_level: float,
    ) -> ProceduralSkill:
        """存储从BG转移来的技能"""
        context_hash = self._hash_context(context)

        # 检查是否已存在，存在则更新
        for existing in self.skills:
            if existing.context_hash == context_hash and existing.action == action:
                existing.motor_pattern = motor_pattern.copy()
                existing.skill_level = max(existing.skill_level, skill_level)
                return existing

        skill = ProceduralSkill(
            context_hash=context_hash,
            action=action,
            motor_pattern=motor_pattern.copy(),
            skill_level=skill_level,
        )
        self.skills.append(skill)

        if len(self.skills) > self.max_skills:
            self.skills.pop(0)

        return skill

    def lookup(self, context: np.ndarray) -> Optional[ProceduralSkill]:
        """查找匹配的技能"""
        context_hash = self._hash_context(context)
        best = None
        for skill in self.skills:
            if skill.context_hash == context_hash:
                if best is None or skill.skill_level > best.skill_level:
                    best = skill
        return best

    def execute_auto(self, context: np.ndarray) -> Optional[Tuple[int, np.ndarray]]:
        """自动执行已存档技能

        Returns:
            (action, motor_output) 或 None (无匹配技能)
        """
        skill = self.lookup(context)
        if skill is None:
            return None

        skill.auto_execution_count += 1
        return (skill.action, skill.motor_pattern)

    def get_summary(self) -> Dict:
        return {
            'total_skills': len(self.skills),
            'total_auto_executions': sum(s.auto_execution_count for s in self.skills),
            'avg_skill_level': np.mean([s.skill_level for s in self.skills]) if self.skills else 0.0,
        }

    @staticmethod
    def _hash_context(context: np.ndarray) -> int:
        discretized = np.sign(context + 0.3) + np.sign(context - 0.3)
        return hash(tuple(discretized.astype(int).tolist()))


class CerebellarPatch(nn.Module):
    """
    小脑模块

    对应小脑的核心功能：
    1. 运动时序学习
    2. 感觉运动映射
    3. 运动协调
    4. 内置错误预测
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 4,
        n_memory_traces: int = 1000,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 细胞层 (Purkinje细胞模拟) - 运动命令输出
        self.purkinje_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        # 攀缘纤维 → 错误信号
        self.climbing_fiber_gain = 0.1

        # 平行纤维 → 工作记忆
        self.parallel_fiber = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 苔藓纤维 → 感觉输入
        self.mossy_fiber = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # 错误预测器
        self.error_predictor = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 记忆痕迹 (时序学习)
        self.memory_traces = deque(maxlen=n_memory_traces)

    def encode_sensory(self, sensory: torch.Tensor) -> torch.Tensor:
        """苔藓纤维编码感觉"""
        return self.mossy_fiber(sensory)

    def compute_motor_command(
        self,
        sensory: torch.Tensor,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算运动命令

        从感觉输入 → 运动输出
        这是小脑的核心功能
        """
        if context is not None:
            # 加入情境信息
            combined = sensory + context * 0.1
        else:
            combined = sensory

        motor_cmd = self.purkinje_layer(combined)
        return motor_cmd

    def predict_error(
        self,
        motor_cmd: torch.Tensor,
    ) -> float:
        """预测错误 (苔藓纤维 → 错误信号)"""
        pred_error = self.error_predictor(motor_cmd)
        if pred_error.numel() > 1:
            pred_error = pred_error.mean()
        return abs(pred_error.item())

    def learn_from_error(
        self,
        sensory: torch.Tensor,
        motor_cmd: torch.Tensor,
        actual_error: float,
    ):
        """
        从错误学习

        模拟攀缘纤维传递的错误信号
        - 调整Purkinje细胞权重
        """
        # 记录记忆痕迹
        self.memory_traces.append({
            'sensory': sensory.detach(),
            'motor_cmd': motor_cmd.detach(),
            'error': actual_error,
        })

        # 如果错误大，触发学习
        if abs(actual_error) > 0.1:
            # 计算误差梯度
            target_cmd = motor_cmd - actual_error * self.climbing_fiber_gain
            loss = F.mse_loss(motor_cmd, target_cmd.detach())

            # 返回loss供优化
            return loss
        return None

    def compute_temporal_sequence(
        self,
        sensory_sequence: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        时序学习

        从一系列感觉输入学习动作序列
        """
        outputs = []
        for sensory in sensory_sequence:
            motor = self.compute_motor_command(sensory)
            outputs.append(motor)
        return outputs


class Cerebellum(nn.Module):
    """
    完整小脑系统

    整合：
    1. 运动学习
    2. 时序整合
    3. 错误预测
    """

    def __init__(
        self,
        sensory_dim: int = 64,
        hidden_dim: int = 64,
        n_motor_joints: int = 4,
    ):
        super().__init__()

        self.sensory_dim = sensory_dim
        self.n_joints = n_motor_joints

        # 程序性记忆 (肌肉记忆存储)
        self.procedural_memory = ProceduralMemory()

        # 多个小脑patch (不同功能区)
        self.motor_patch = CerebellarPatch(
            input_dim=sensory_dim,
            hidden_dim=hidden_dim,
            output_dim=n_motor_joints,
        )

        self.coordination_patch = CerebellarPatch(
            input_dim=sensory_dim,
            hidden_dim=hidden_dim,
            output_dim=n_motor_joints,
        )

        # 时序学习器
        self.temporal_learning = nn.LSTM(
            input_size=sensory_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.sequence_output = nn.Linear(hidden_dim, n_motor_joints)

        # 统计
        self.learning_steps = 0
        self.total_error = 0.0

    def forward(
        self,
        sensory: torch.Tensor,
        context: torch.Tensor = None,
        mode: str = "motor",
    ) -> Dict:
        """
        处理运动控制

        Args:
            sensory: 感觉输入
            context: 情境信息
            mode: "motor" | "coordination" | "sequence"

        Returns:
            motor_command: 运动指令
            predicted_error: 预测误差
        """
        # 编码感觉
        encoded = self.motor_patch.encode_sensory(sensory)

        if mode == "motor":
            motor_cmd = self.motor_patch.compute_motor_command(
                encoded,
                context
            )
        elif mode == "coordination":
            motor_cmd = self.coordination_patch.compute_motor_command(
                encoded,
                context
            )
        elif mode == "sequence":
            # 时序模式
            if sensory.dim() == 2:
                sensory = sensory.unsqueeze(1)  # [batch, seq, dim]
            lstm_out, _ = self.temporal_learning(sensory)
            motor_cmd = self.sequence_output(lstm_out[:, -1])
        else:
            motor_cmd = torch.zeros(self.n_joints)

        # 预测误差
        pred_error = self.motor_patch.predict_error(motor_cmd)

        return {
            'motor_command': motor_cmd,
            'predicted_error': pred_error,
        }

    def learn(
        self,
        sensory: torch.Tensor,
        motor_cmd: torch.Tensor,
        actual_error: float,
    ) -> Dict:
        """
        从实际错误学习
        """
        loss = self.motor_patch.learn_from_error(
            sensory,
            motor_cmd,
            actual_error
        )

        self.learning_steps += 1
        self.total_error += actual_error

        return {
            'loss': loss,
            'avg_error': self.total_error / self.learning_steps,
        }

    def receive_archived_skill(
        self,
        context: np.ndarray,
        action: int,
        motor_pattern: np.ndarray,
        skill_level: float,
    ) -> None:
        """接收BG存档的技能 (写入程序性记忆)

        对应"熟能生巧"的技能转移：
        BG判断某个动作已足够熟练 → 转移到小脑自动执行
        """
        self.procedural_memory.store(
            context=context,
            action=action,
            motor_pattern=motor_pattern,
            skill_level=skill_level,
        )
        print(f"[CEREBELLUM] Skill received from BG: action={action}, "
              f"level={skill_level:.2f}, total_procedural={len(self.procedural_memory.skills)}")

    def execute_procedural(
        self,
        context: np.ndarray,
    ) -> Optional[Dict]:
        """执行程序性记忆 (自动执行已存档技能)

        Returns:
            None if no matching skill, else {action, motor_output, is_automatic}
        """
        result = self.procedural_memory.execute_auto(context)
        if result is None:
            return None

        action, motor_pattern = result
        return {
            'action': action,
            'motor_output': motor_pattern,
            'is_automatic': True,  # 标记为自动执行 (无需意识参与)
        }

    def get_statistics(self) -> Dict:
        """获取统计"""
        return {
            'learning_steps': self.learning_steps,
            'avg_error': self.total_error / max(1, self.learning_steps),
        }


# ============ 小脑-脊髓协同系统 ============

class CerebelloSpinalCoordination(nn.Module):
    """
    小脑-脊髓协同系统

    核心协同机制：
    1. 小脑学习运动程序 → 下行到脊髓CPG
    2. 脊髓执行反射 → 上行反馈到小脑
    3. 闭环运动学习

    架构：
    [Cerebellum] ←→ [Spinal Cord]
         ↓              ↓
      学习           执行
         ↓              ↓
      [Motor Command]
    """

    def __init__(
        self,
        sensory_dim: int = 64,
        n_joints: int = 4,
    ):
        super().__init__()

        # 小脑
        self.cerebellum = Cerebellum(
            sensory_dim=sensory_dim,
            n_motor_joints=n_joints,
        )

        # 脊髓
        self.spinal = SpinalCord(
            n_joints=n_joints,
            cpg_frequency=2.0,
        )

        # 学习控制器
        self.learning_rate = 0.01
        self.error_history = deque(maxlen=100)

    def forward(
        self,
        sensory: torch.Tensor,
        execute: bool = True,
    ) -> Dict:
        """
        协同控制

        Args:
            sensory: 感觉输入
            execute: 是否执行运动

        Returns:
            result: 执行结果
        """
        # 1. 小脑计算运动命令
        cereb_result = self.cerebellum(sensory)
        motor_cmd = cereb_result['motor_command']
        pred_error = cereb_result['predicted_error']

        if not execute:
            return {
                'motor_command': motor_cmd,
                'predicted_error': pred_error,
                'executed': False,
            }

        # 2. 脊髓执行
        motor_cmd_np = motor_cmd.detach().numpy()
        reflex_input = None

        # 是否有危险输入
        if pred_error > 0.5:
            reflex_input = np.array([pred_error])

        spinal_result = self.spinal.forward(
            cpg_command=False,  # 使用学习的命令而非CPG
            reflex_input=reflex_input,
        )

        # 3. 收集执行结果
        proprio = self.spinal.get_proprioception()

        # 4. 计算实际误差 (vs 预期)
        actual_error = abs(spinal_result.muscle_activations.mean())

        # 5. 更新误差历史
        self.error_history.append(actual_error)

        return {
            'motor_command': motor_cmd,
            'predicted_error': pred_error,
            'actual_error': actual_error,
            'proprioception': proprio,
            'executed': True,
        }

    def learn_motor_program(
        self,
        sensory: torch.Tensor,
        desired_output: torch.Tensor,
    ) -> float:
        """
        学习运动程序

        监督学习：给定期望输出 → 学习映射
        """
        # 小脑产生命令
        result = self.cerebellum(sensory)
        motor_cmd = result['motor_command']

        # 计算误差
        error = F.mse_loss(motor_cmd, desired_output)

        # 学习
        self.cerebellum.learn(sensory, motor_cmd, error.item())

        return error.item()

    def learn_reflex(
        self,
        sensory: torch.Tensor,
        unexpected_disturbance: float,
    ) -> Dict:
        """
        从意外干扰学习反射

        强化学习：学会对特定干扰的反射
        """
        # 模拟干扰
        response = self.spinal.reflex.compute_withdrawal_reflex(
            unexpected_disturbance
        )

        # 记录
        self.error_history.append(unexpected_disturbance)

        return {
            'reflex_response': response,
            'avg_disturbance': np.mean(list(self.error_history)),
        }

    def adapt_to_error(
        self,
        sensory: torch.Tensor,
        actual_error: float,
    ):
        """
        误差适应

        从执行误差学习调整
        """
        cereb_result = self.cerebellum(sensory)
        motor_cmd = cereb_result['motor_command']

        # 从实际误差学习
        self.cerebellum.learn(sensory, motor_cmd, actual_error)

    def enable_cpg_walk(self):
        """启用CPG行走模式"""
        self.spinal.cpg.activate()
        # 设置典型步态相位
        offsets = [0, np.pi/2, np.pi, 3*np.pi/2]
        for i, offset in enumerate(offsets):
            self.spinal.cpg.set_phase_offset(i, offset)

    def disable_cpg(self):
        """停用CPG"""
        self.spinal.cpg.deactivate()

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            'cerebellum': self.cerebellum.get_statistics(),
            'is_cpg_active': self.spinal.cpg.is_active,
            'avg_error': np.mean(list(self.error_history)) if self.error_history else 0,
        }


# ============ 便捷函数 ============

def create_cerebello_spinal(
    sensory_dim: int = 64,
    n_joints: int = 4,
) -> CerebelloSpinalCoordination:
    """创建小脑-脊髓协同系统"""
    return CerebelloSpinalCoordination(
        sensory_dim=sensory_dim,
        n_joints=n_joints,
    )


__all__ = [
    'MotorCommand',
    'ReflexResponse',
    'CentralPatternGenerator',
    'ReflexPathway',
    'SpinalCord',
    'ProceduralSkill',
    'ProceduralMemory',
    'CerebellarPatch',
    'Cerebellum',
    'CerebelloSpinalCoordination',
    'create_cerebello_spinal',
]