"""
对话情绪响应系统

基于Simulacrum的心理学期制
实现具有情绪感知和同理心的对话AI
"""
import torch
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class EmotionalDialogue:
    """
    情绪感知对话系统
    """
    def __init__(self):
        base = 'core/language_cortex.py'
        self.lang = load(base, 'language_cortex')
        self.model = self.lang.create_language_cortex(vocab_size=1000, use_parallel=False)
        self.model.eval()

        # 对话历史
        self.history = []
        self.max_history = 10

    def process_input(self, user_input: str):
        """处理用户输入"""
        # 分词
        tokens = [ord(c) % 1000 for c in user_input.lower()[:16]]
        tokens = tokens + [0] * (16 - len(tokens))
        tokens = torch.tensor([tokens])

        with torch.no_grad():
            result = self.model(tokens, return_emotion=True)

        # 提取信息
        emotion = result.get('emotion_state', {})
        valence = emotion.get('valence', 0)
        arousal = emotion.get('arousal', 0)

        # 存储
        self.history.append({
            'input': user_input,
            'emotion': emotion,
        })

        return {
            'valence': valence,
            'arousal': arousal,
            'features': result['features'],
        }

    def generate_response(self, user_input: str) -> str:
        """生成情绪响应"""
        # 处理输入
        info = self.process_input(user_input)

        # 基于情绪选择响应策略
        v, a = info['valence'].item(), info['arousal'].item()

        # 响应策略映射
        if v < -0.3 and a > 0.5:
            # 负面+高唤醒 → 安慰
            responses = [
                "我理解你的感受，一起想办法解决吧",
                "别担心，情况会慢慢变好的",
                "我在这里陪你",
            ]
        elif v < -0.3:
            # 负面+低唤醒 → 同理
            responses = [
                "我能感受到你的难过",
                "这确实让人沮丧",
            ]
        elif a > 0.7:
            # 高唤醒 → 平静化
            responses = [
                "慢慢说，不着急",
                "深呼吸，我们一起理清思路",
            ]
        elif v > 0.3:
            # 正面 → 积极回应
            responses = [
                "太好了，继续保持!",
                "你做得很好!",
                "真棒!",
            ]
        else:
            # 中性 → 继续对话
            responses = [
                "然后呢?",
                "我听着",
                "嗯嗯",
            ]

        # 应用元认知调整
        thought = info['features']
        strategy, _ = self.model.metacognition.self_regulate(thought, 'reasoning')

        # 选择响应
        import random
        response = random.choice(responses)

        # 添加元认知调整
        if strategy == 'critical':
            response = f"让我想想... {response}"
        elif strategy == 'creative':
            response = f"有个新想法... {response}"

        # 记录
        self.history.append({
            'input': user_input,
            'response': response,
            'strategy': strategy,
        })

        return response

    def get_empathy_score(self) -> float:
        """计算同理心得分"""
        if len(self.history) < 2:
            return 0.5

        # 检查情绪匹配
        matches = 0
        for i in range(1, len(self.history)):
            prev_emo = self.history[i-1].get('emotion', {})
            resp = self.history[i].get('response', '')

            # 简单检查
            if '别' in resp or '我理解' in resp:
                matches += 0.5
            elif prev_emo.get('valence', 0) < 0 and '难过' in resp:
                matches += 0.5

        return matches / len(self.history)


def demo():
    """演示"""
    print("=" * 40)
    print("情绪感知对话系统演示")
    print("=" * 40)

    dialog = EmotionalDialogue()

    test_inputs = [
        "i am so happy today",
        "this makes me angry",
        "i feel scared about the exam",
        "everything is fine",
    ]

    for inp in test_inputs:
        resp = dialog.generate_response(inp)
        print(f"\n用户: {inp}")
        print(f"AI: {resp}")

        info = dialog.history[-1]
        print(f"情绪: v={info.get('emotion',{}).get('valence',0):.2f}, a={info.get('emotion',{}).get('arousal',0):.2f}")

    print(f"\n同理心得分: {dialog.get_empathy_score():.2f}")


if __name__ == "__main__":
    demo()