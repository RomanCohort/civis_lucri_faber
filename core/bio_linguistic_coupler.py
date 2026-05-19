"""
生物-语言耦合模块 (Bio-Linguistic Coupler)

核心设计理念：神经系统状态直接决定语言输出，而非仅作为提示词建议。

生物-语言耦合的四个层次：
1. 词汇层 (Lexical)    — 神经递质决定词汇选择
2. 句法层 (Syntactic)  — 激素水平决定句子结构
3. 语义层 (Semantic)   — 唤醒度决定话题深度
4. 韵律层 (Prosodic)   — 压力/疲劳决定停顿和语调

数据流：
  LLM Output Text
       ↓
  BioLinguisticCoupler.process()
       ↓
  ┌─ BioLexicalSelector    (词汇调整)
  ├─ BioSyntacticPlanner   (句法调整)
  ├─ BioSemanticFilter     (语义过滤/增强)
  ├─ BioEmotionalTone     (情感色调)
  └─ BioHesitationMarker   (犹豫标记注入)
       ↓
  Modified Text
       ↓
  VocalSystem (韵律生成)

生物学对应：
  - 前额叶皮层 (PFC) → 词汇选择抑制/激活
  - Broca区 → 句法结构规划
  - 颞叶 → 语义整合
  - 边缘系统 → 情感语调
  - 基底神经节 → 言语启动/犹豫
"""

import re
import random
from typing import Dict, List, Optional, Tuple
import numpy as np


# ============ 词汇表：按情绪/状态分类 ============

class LexicalCategory:
    """情绪/状态词汇类别"""

    # 高多巴胺（探索、创新、积极）
    DOPAMINE_RICH = {
        'adjectives': ['精彩的', '令人兴奋的', '美妙的', '不可思议的', '全新的', '独特的',
                       '非凡的', '惊人的', '迷人的', '引人入胜的'],
        'verbs': ['探索', '发现', '创造', '想象', '发现', '突破', '创新', '尝试',
                  '冒险', '挑战', '发现', '挖掘', '拓展'],
        'nouns': ['可能性', '机会', '创意', '灵感', '探索', '冒险', '新视角', '惊喜'],
        'adverbs': ['充满热情地', '兴致勃勃地', '迫不及待地', '积极地', '勇敢地'],
    }

    # 低多巴胺（保守、退缩、消极）
    DOPAMINE_LOW = {
        'adjectives': ['平淡的', '普通的', '一般的', '常见的', '老旧的', '无聊的',
                       '乏味的', '没什么特别的', '普通的'],
        'verbs': ['维持', '保持', '观望', '等待', '跟随', '接受', '应付', '处理'],
        'nouns': ['现状', '常规', '平淡', '例行公事'],
        'adverbs': ['一般地', '平常地', '平淡地', '勉强地'],
    }

    # 高皮质醇（压力、紧张）
    CORTISOL_HIGH = {
        'adjectives': ['紧迫的', '严重的', '麻烦的', '棘手的', '不确定的', '困难的',
                       '危险的', '紧急的', '关键的', '重要的'],
        'verbs': ['处理', '应对', '解决', '面对', '承担', '担心', '焦虑', '紧张',
                  '抓紧', '赶'],
        'nouns': ['压力', '问题', '困难', '挑战', '危机', '紧急情况'],
        'adverbs': ['紧张地', '急切地', '匆忙地', '焦急地', '紧迫地'],
    }

    # 低皮质醇（放松、平静）
    CORTISOL_LOW = {
        'adjectives': ['轻松的', '悠闲的', '舒适的', '平静的', '自在的', '随意的'],
        'verbs': ['享受', '放松', '闲聊', '分享', '交流', '陪伴'],
        'nouns': ['休闲', '舒适', '平静', '安宁'],
        'adverbs': ['轻松地', '悠闲地', '慢慢地', '随意地', '从容地'],
    }

    # 高唤醒（兴奋、激动）
    AROUSAL_HIGH = {
        'adjectives': ['激动的', '兴奋的', '热情的', '激动的', '热烈的', '强烈的'],
        'verbs': ['激动', '兴奋', '热情', '激动地', '强烈地', '热烈地表达'],
        'nouns': ['热情', '激动', '兴奋'],
        'adverbs': ['激动地', '兴奋地', '热情地', '热烈地'],
    }

    # 低唤醒（疲惫、困倦）
    AROUSAL_LOW = {
        'adjectives': ['疲惫的', '困倦的', '无力的', '迟缓的', '萎靡的'],
        'verbs': ['躺着', '休息', '犯困', '想睡觉', '没力气'],
        'nouns': ['疲惫', '困意', '倦意', '无力'],
        'adverbs': ['疲惫地', '慢吞吞地', '有气无力地', '懒洋洋地', '昏昏沉沉地'],
    }

    # 恐惧状态
    FEAR = {
        'adjectives': ['可怕的', '危险的', '令人担忧的', '不确定的', '令人不安的'],
        'verbs': ['害怕', '担心', '恐惧', '忧虑', '警惕', '躲避'],
        'nouns': ['恐惧', '担忧', '危险', '不安', '威胁'],
        'adverbs': ['害怕地', '担忧地', '不安地', '惊恐地'],
    }

    # 愤怒状态
    ANGER = {
        'adjectives': ['可恶的', '过分的', '令人愤怒的', '荒谬的', '不可接受的'],
        'verbs': ['愤怒', '生气', '恼火', '不满', '抗议', '反对', '指责'],
        'nouns': ['愤怒', '怒火', '不满', '怨气'],
        'adverbs': ['愤怒地', '生气地', '恼火地', '激动地'],
    }

    # 喜悦状态
    JOY = {
        'adjectives': ['美好的', '开心的', '幸福的', '愉快的', '欢乐的', '快乐的'],
        'verbs': ['开心', '高兴', '快乐', '欢乐', '欢庆', '庆祝'],
        'nouns': ['快乐', '幸福', '欢乐', '愉快', '喜悦'],
        'adverbs': ['开心地', '高兴地', '快乐地', '愉快地', '欢天喜地地'],
    }

    # 悲伤状态
    SADNESS = {
        'adjectives': ['伤心的', '难过的', '沮丧的', '失落的', '忧郁的', '痛苦的'],
        'verbs': ['伤心', '难过', '失落', '沮丧', '痛苦', '悲伤', '哀伤'],
        'nouns': ['悲伤', '难过', '失落', '沮丧', '痛苦', '哀伤'],
        'adverbs': ['难过地', '伤心欲绝地', '沮丧地', '忧郁地'],
    }

    # 血清素低（情绪不稳定）
    SEROTONIN_LOW = {
        'adjectives': ['矛盾的', '复杂的', '混乱的', '模糊的', '不确定的', '摇摆不定的'],
        'verbs': ['犹豫', '矛盾', '纠结', '徘徊', '彷徨', '动摇'],
        'nouns': ['矛盾', '纠结', '混乱', '模糊', '不确定'],
        'adverbs': ['犹豫地', '矛盾地', '纠结地', '摇摆地'],
    }

    # 催产素高（社交、信任）
    OXYTOCIN_HIGH = {
        'adjectives': ['温暖的', '亲切的', '友善的', '关心的', '信任的', '亲近的'],
        'verbs': ['关心', '照顾', '陪伴', '倾听', '理解', '信任', '依赖'],
        'nouns': ['关怀', '温暖', '信任', '友情', '亲近', '陪伴'],
        'adverbs': ['温暖地', '亲切地', '关心地', '友善地'],
    }


# ============ 犹豫标记（基于基底神经节活动） ============

class HesitationMarkers:
    """犹豫标记词（模拟基底神经节的言语启动延迟）"""

    # 轻度犹豫
    LIGHT = ['嗯', '呃', '这个', '那个', '怎么说呢', '让我想想']

    # 中度犹豫
    MEDIUM = ['嗯...', '呃...', '嗯嗯...', '这个嘛...', '怎么说呢...',
              '让我想想...', '有点...', '好像...']

    # 重度犹豫/不确定
    HEAVY = ['嗯...嗯...', '呃...这个...', '我也不太确定...', '好像...又好像...',
             '这个有点复杂...', '让我好好想想...', '不太确定该怎么表达...',
             '嗯...可能...', '这...这很难说...']

    # 填充词（无意义但自然）
    FILLERS = ['就是说', '你知道', '然后呢', '反正就是', '总之', '就是说啊',
               '对吧', '差不多', '大概', '应该', '可能吧']


# ============ 句法规则 ============

class SyntacticRules:
    """句法结构调整规则"""

    @staticmethod
    def split_long_sentences(text: str, max_length: int = 30) -> str:
        """将长句拆分为短句（高皮质醇/疲劳时）"""
        sentences = re.split(r'([。！？；\n])', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''
            words = sent.strip()
            if not words:
                continue

            word_count = len(words)
            if word_count > max_length:
                # 拆分长句：在逗号处拆分
                parts = re.split(r'，', words)
                if len(parts) > 1:
                    new_sent = '。'.join([p.strip() for p in parts if p.strip()])
                    result.append(new_sent)
                    result.append(punct)
                else:
                    result.append(words)
                    result.append(punct)
            else:
                result.append(words)
                result.append(punct)

        return ''.join(result)

    @staticmethod
    def simplify_sentences(text: str, level: float = 0.5) -> str:
        """简化句子结构（level 0-1，越高越简单）"""
        if level < 0.3:
            return text

        # 减少从句连接词
        simplifications = [
            (r'，因此', '，所以'),
            (r'，然而', '，但是'),
            (r'，虽然', '，但'),
            (r'，因为', '，由于'),
            (r'，但是', '，但'),
            (r'，如果', '，要是'),
        ]

        result = text
        for old, new in simplifications:
            result = re.sub(old, new, result)

        # 减少复杂修饰
        if level > 0.6:
            # 移除大量形容词
            result = re.sub(r'（[^）]*）', '', result)  # 括号内容

        return result

    @staticmethod
    def inject_sentence_fragments(text: str, level: float = 0.5) -> str:
        """注入句子碎片（高疲劳时）"""
        if level < 0.5:
            return text

        # 在句子中间插入碎片
        fragments = ['...', '然后...', '那个...', '就是说...']
        sentences = re.split(r'([。！？\n])', text)

        result = []
        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i]
            punct = sentences[i + 1]
            if len(sent) > 10 and random.random() < level * 0.3:
                result.append(sent)
                result.append(random.choice(fragments))
                result.append(punct)
            else:
                result.append(sent)
                result.append(punct)

        return ''.join(result)


# ============ 主类：生物-语言耦合器 ============

class BioLinguisticCoupler:
    """
    生物-语言耦合器

    直接将神经系统状态映射为语言输出特征。
    工作在 LLM 输出和发声系统之间，对文本进行生物驱动的后处理。

    用法：
        coupler = BioLinguisticCoupler()
        modified_text = coupler.process(
            text="LLM生成的原始回复",
            bio_state={
                'dopamine': 0.7, 'serotonin': 0.4, 'cortisol': 0.6,
                'fatigue': 0.5, 'arousal': 0.6, 'valence': 0.3,
                'defense': '', 'oxytocin': 0.5, 'emotion': 'neutral'
            }
        )
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.lexical = LexicalCategory()
        self.hesitation = HesitationMarkers()
        self.syntactic = SyntacticRules()

    def process(self, text: str, bio_state: Dict) -> str:
        """
        对文本进行生物驱动的后处理

        Args:
            text: LLM 生成的原始文本
            bio_state: 生物状态字典

        Returns:
            修改后的文本
        """
        if not text or not text.strip():
            return text

        result = text

        # 1. 情感色调注入（最优先）
        result = self._apply_emotional_tone(result, bio_state)

        # 2. 句法结构调整
        result = self._apply_syntactic_modulation(result, bio_state)

        # 3. 犹豫标记注入
        result = self._inject_hesitation_markers(result, bio_state)

        # 4. 词汇替换（基于神经递质）
        result = self._apply_lexical_replacement(result, bio_state)

        # 5. 句子长度调整
        result = self._adjust_sentence_length(result, bio_state)

        # 6. 语义深度过滤
        result = self._apply_semantic_filtering(result, bio_state)

        return result

    def _apply_emotional_tone(self, text: str, bio_state: Dict) -> str:
        """根据主导情绪调整文本的情感色调"""
        emotion = bio_state.get('emotion', 'neutral')
        valence = bio_state.get('valence', 0.0)
        arousal = bio_state.get('arousal', 0.5)

        if emotion == 'neutral' and abs(valence) < 0.2:
            return text

        # 选择情绪词汇表
        emotion_words = None
        if emotion == 'fear' or (valence < -0.3 and arousal > 0.5):
            emotion_words = self.lexical.FEAR
        elif emotion == 'anger' or valence < -0.5:
            emotion_words = self.lexical.ANGER
        elif emotion == 'joy' or valence > 0.3:
            emotion_words = self.lexical.JOY
        elif emotion == 'sadness' or valence < -0.3:
            emotion_words = self.lexical.SADNESS

        if emotion_words is None:
            return text

        # 在适当位置注入情绪词汇
        sentences = re.split(r'([。！？\n])', text)
        result_parts = []
        word_pool = (emotion_words['adjectives'] + emotion_words['verbs'] +
                     emotion_words['adverbs'])

        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''

            if sent.strip() and random.random() < 0.3 * abs(valence):
                word = random.choice(word_pool)
                # 在句首插入情绪词
                sent = f"{word}，{sent.strip()}"

            result_parts.append(sent)
            if punct:
                result_parts.append(punct)

        return ''.join(result_parts)

    def _apply_syntactic_modulation(self, text: str, bio_state: Dict) -> str:
        """根据皮质醇和疲劳调整句子结构"""
        cortisol = bio_state.get('cortisol', 0.3)
        fatigue = bio_state.get('fatigue', 0.3)

        # 高皮质醇 → 短句
        if cortisol > 0.5:
            max_len = max(15, int(40 - cortisol * 30))
            text = self.syntactic.split_long_sentences(text, max_length=max_len)

        # 高疲劳 → 简化结构
        if fatigue > 0.4:
            simplify_level = min(1.0, (fatigue - 0.4) * 1.5)
            text = self.syntactic.simplify_sentences(text, level=simplify_level)

        # 极高疲劳 → 句子碎片化
        if fatigue > 0.6:
            fragment_level = (fatigue - 0.6) * 2.0
            text = self.syntactic.inject_sentence_fragments(text, level=fragment_level)

        return text

    def _inject_hesitation_markers(self, text: str, bio_state: Dict) -> str:
        """注入犹豫标记（模拟基底神经节言语启动延迟）"""
        cortisol = bio_state.get('cortisol', 0.3)
        fatigue = bio_state.get('fatigue', 0.3)
        serotonin = bio_state.get('serotonin', 0.5)
        arousal = bio_state.get('arousal', 0.5)

        # 计算犹豫概率
        hesitation_prob = 0.0
        hesitation_level = 'LIGHT'

        if fatigue > 0.5:
            hesitation_prob += (fatigue - 0.5) * 0.5
            hesitation_level = 'MEDIUM'
        if cortisol > 0.6:
            hesitation_prob += (cortisol - 0.6) * 0.4
            hesitation_level = 'HEAVY'
        if serotonin < 0.4:
            hesitation_prob += (0.4 - serotonin) * 0.3
            hesitation_level = 'MEDIUM'
        if arousal < 0.3:
            hesitation_prob += (0.3 - arousal) * 0.3
            hesitation_level = 'HEAVY'

        if hesitation_prob < 0.05:
            return text

        # 在句子开头注入犹豫标记
        sentences = re.split(r'([。！？\n,，])', text)
        result_parts = []

        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''

            if sent.strip() and random.random() < hesitation_prob:
                if hesitation_level == 'LIGHT':
                    marker = random.choice(self.hesitation.LIGHT)
                elif hesitation_level == 'MEDIUM':
                    marker = random.choice(self.hesitation.MEDIUM)
                else:
                    marker = random.choice(self.hesitation.HEAVY)

                sent = f"{marker}，{sent.strip()}"

            result_parts.append(sent)
            if punct:
                result_parts.append(punct)

        # 随机注入填充词
        filler_prob = hesitation_prob * 0.3
        if filler_prob > 0.05:
            fillers = self.hesitation.FILLERS
            result_parts_text = ''.join(result_parts)

            for filler in fillers:
                if random.random() < filler_prob:
                    # 在逗号处随机插入填充词
                    result_parts_text = re.sub(
                        r'，',
                        lambda m: '，' + random.choice(fillers) if random.random() < filler_prob else '，',
                        result_parts_text,
                        count=1
                    )

            return result_parts_text

        return ''.join(result_parts)

    def _apply_lexical_replacement(self, text: str, bio_state: Dict) -> str:
        """根据神经递质替换词汇"""
        dopamine = bio_state.get('dopamine', 0.5)
        serotonin = bio_state.get('serotonin', 0.5)
        cortisol = bio_state.get('cortisol', 0.3)
        oxytocin = bio_state.get('oxytocin', 0.5)

        words = text.split()

        for i, word in enumerate(words):
            # 多巴胺影响形容词/副词
            if random.random() < 0.15:
                if dopamine > 0.6:
                    replacement = self._replace_with_category(word, self.lexical.DOPAMINE_RICH)
                elif dopamine < 0.3:
                    replacement = self._replace_with_category(word, self.lexical.DOPAMINE_LOW)
                else:
                    replacement = None

                if replacement:
                    words[i] = replacement

                # 皮质醇影响动词
            if cortisol > 0.6 and random.random() < 0.15:
                replacement = self._replace_with_category(word, self.lexical.CORTISOL_HIGH)
                if replacement:
                    words[i] = replacement
            elif cortisol < 0.3 and random.random() < 0.1:
                replacement = self._replace_with_category(word, self.lexical.CORTISOL_LOW)
                if replacement:
                    words[i] = replacement

            # 催产素影响情感词
            if oxytocin > 0.6 and random.random() < 0.1:
                replacement = self._replace_with_category(word, self.lexical.OXYTOCIN_HIGH)
                if replacement:
                    words[i] = replacement

        return ''.join(words)

    def _replace_with_category(self, word: str, category: Dict) -> Optional[str]:
        """从类别中随机选择一个词替换"""
        for pos_list in category.values():
            if pos_list and random.random() < 0.3:
                return random.choice(pos_list)
        return None

    def _adjust_sentence_length(self, text: str, bio_state: Dict) -> str:
        """根据生物状态调整句子长度"""
        cortisol = bio_state.get('cortisol', 0.3)
        fatigue = bio_state.get('fatigue', 0.3)
        arousal = bio_state.get('arousal', 0.5)

        sentences = re.split(r'([。！？\n])', text)

        # 高唤醒 + 高皮质醇 → 句子更短
        if cortisol > 0.5 and arousal > 0.5:
            for i in range(0, len(sentences), 2):
                sent = sentences[i]
                if len(sent) > 20:
                    # 截断长句
                    sentences[i] = sent[:20] + '...'

        # 高疲劳 → 减少总句子数（合并）
        if fatigue > 0.7:
            short_sentences = [s for s in sentences if len(s.strip()) > 0 and len(s) < 5]
            if short_sentences:
                # 移除过短的句子碎片
                result = []
                for s in sentences:
                    if len(s.strip()) > 0 and len(s) >= 3:
                        result.append(s)
                sentences = result

        return ''.join(sentences)

    def _apply_semantic_filtering(self, text: str, bio_state: Dict) -> str:
        """根据唤醒度和情绪调整语义内容深度"""
        arousal = bio_state.get('arousal', 0.5)
        fatigue = bio_state.get('fatigue', 0.3)
        dopamine = bio_state.get('dopamine', 0.5)

        # 高疲劳 + 低唤醒 → 移除复杂解释
        if fatigue > 0.6 and arousal < 0.4:
            # 移除括号中的详细解释
            text = re.sub(r'（[^）]{50,}）', '（略）', text)
            # 移除长修饰成分
            text = re.sub(r'，[^，]{30,}，', '，', text)

        # 低多巴胺 → 移除探索性/创新性表达
        if dopamine < 0.3:
            exploratory_patterns = [
                r'也许可以',
                r'不妨',
                r'试试看',
                r'大胆地',
                r'探索一下',
                r'或许',
            ]
            for pattern in exploratory_patterns:
                if random.random() < 0.5:
                    text = re.sub(pattern, '', text)

        # 低唤醒 → 聚焦核心信息，移除发散内容
        if arousal < 0.3:
            # 移除反问句（需要更多认知资源）
            text = re.sub(r'不是吗[？?]?', '。', text)
            text = re.sub(r'对吧[？?]?', '。', text)
            # 移除感叹
            text = re.sub(r'！{2,}', '！', text)

        return text

    def get_prosodic_params(self, bio_state: Dict) -> Dict:
        """
        生成韵律参数（供 VocalSystem 使用）

        Returns:
            韵律参数字典: {
                'speech_rate': 语速 (0.5-2.0),
                'pitch_baseline': 基频 (Hz),
                'pitch_range': 音高范围 (semitones),
                'pause_frequency': 停顿频率 (0-1),
                'pause_duration': 停顿时长 (sec),
                'volume': 响度 (0-1),
                'voice_register': 'modal'/'fry'/'breathy',
                'articulation_clarity': 清晰度 (0-1),
                'jitter': 微抖动 (0-0.05)
            }
        """
        arousal = bio_state.get('arousal', 0.5)
        fatigue = bio_state.get('fatigue', 0.3)
        dopamine = bio_state.get('dopamine', 0.5)
        norepinephrine = bio_state.get('norepinephrine', 0.3)
        cortisol = bio_state.get('cortisol', 0.3)
        serotonin = bio_state.get('serotonin', 0.5)
        valence = bio_state.get('valence', 0.0)
        heart_rate = bio_state.get('heart_rate', 72.0)
        respiratory_rate = bio_state.get('respiratory_rate', 12.0)

        # 语速：唤醒度高快，疲劳时慢
        speech_rate = 0.7 + arousal * 0.6 - fatigue * 0.4
        speech_rate = np.clip(speech_rate, 0.4, 2.0)

        # 基频：去甲肾上腺素+唤醒度决定音调
        # 高NE+高唤醒 = 高音调；低唤醒 = 低音调
        pitch_baseline = 120 + arousal * 80 + norepinephrine * 40 - fatigue * 30
        pitch_baseline = np.clip(pitch_baseline, 80, 250)

        # 音高范围（表达性）
        pitch_range = 3 + valence * 10 + arousal * 7 - fatigue * 5
        pitch_range = np.clip(pitch_range, 0, 20)

        # 停顿频率：疲劳/低唤醒时更多停顿
        pause_frequency = fatigue * 0.4 + (1 - arousal) * 0.3 + serotonin * 0.1
        pause_frequency = np.clip(pause_frequency, 0, 1)

        # 停顿时长：低唤醒时更长
        pause_duration = 0.3 + (1 - arousal) * 0.5 + fatigue * 0.3
        pause_duration = np.clip(pause_duration, 0.1, 2.0)

        # 响度：唤醒度高+多巴胺高 = 响亮；疲劳 = 轻柔
        volume = 0.4 + arousal * 0.3 + dopamine * 0.2 - fatigue * 0.3
        volume = np.clip(volume, 0.1, 1.0)

        # 嗓音类型
        if fatigue > 0.7 or arousal < 0.2:
            voice_register = 'fry'  # 气泡音（极度疲劳）
        elif arousal > 0.7 and valence > 0.3:
            voice_register = 'breathy'  # 气声（兴奋/开心）
        else:
            voice_register = 'modal'  # 正常

        # 发音清晰度：皮质醇高/疲劳时降低
        articulation_clarity = 1.0 - cortisol * 0.3 - fatigue * 0.4 - norepinephrine * 0.2
        articulation_clarity = np.clip(articulation_clarity, 0.3, 1.0)

        # 微抖动（紧张时）
        jitter = 0.005 + cortisol * 0.02 + arousal * 0.01
        jitter = np.clip(jitter, 0.0, 0.05)

        return {
            'speech_rate': round(speech_rate, 3),
            'pitch_baseline': round(pitch_baseline, 1),
            'pitch_range': round(pitch_range, 1),
            'pause_frequency': round(pause_frequency, 3),
            'pause_duration': round(pause_duration, 3),
            'volume': round(volume, 3),
            'voice_register': voice_register,
            'articulation_clarity': round(articulation_clarity, 3),
            'jitter': round(jitter, 4),
            'breathiness': round(0.2 + arousal * 0.3 if voice_register == 'breathy' else fatigue * 0.2, 3),
        }


# ============ 快速测试 ============

if __name__ == '__main__':
    coupler = BioLinguisticCoupler()

    # 测试用例
    test_text = "我觉得这个问题很有意思，可以从多个角度来分析。首先，我们需要考虑整体的环境因素，然后才能得出一个合理的结论。你觉得呢？"

    test_cases = [
        {'emotion': 'neutral', 'dopamine': 0.5, 'cortisol': 0.3, 'fatigue': 0.3,
         'arousal': 0.5, 'serotonin': 0.5, 'oxytocin': 0.5, 'valence': 0.0,
         'norepinephrine': 0.3, 'heart_rate': 72, 'respiratory_rate': 12,
         'defense': '', 'name': '正常状态'},

        {'emotion': 'fear', 'dopamine': 0.3, 'cortisol': 0.8, 'fatigue': 0.4,
         'arousal': 0.7, 'serotonin': 0.3, 'oxytocin': 0.3, 'valence': -0.5,
         'norepinephrine': 0.8, 'heart_rate': 95, 'respiratory_rate': 16,
         'defense': '', 'name': '恐惧+高压'},

        {'emotion': 'joy', 'dopamine': 0.9, 'cortisol': 0.1, 'fatigue': 0.2,
         'arousal': 0.9, 'serotonin': 0.7, 'oxytocin': 0.6, 'valence': 0.8,
         'norepinephrine': 0.6, 'heart_rate': 85, 'respiratory_rate': 14,
         'defense': '', 'name': '兴奋+高多巴胺'},

        {'emotion': 'neutral', 'dopamine': 0.2, 'cortisol': 0.2, 'fatigue': 0.8,
         'arousal': 0.2, 'serotonin': 0.3, 'oxytocin': 0.4, 'valence': -0.2,
         'norepinephrine': 0.2, 'heart_rate': 58, 'respiratory_rate': 10,
         'defense': '', 'name': '极度疲劳'},

        {'emotion': 'anger', 'dopamine': 0.4, 'cortisol': 0.7, 'fatigue': 0.3,
         'arousal': 0.9, 'serotonin': 0.4, 'oxytocin': 0.2, 'valence': -0.6,
         'norepinephrine': 0.9, 'heart_rate': 110, 'respiratory_rate': 18,
         'defense': 'fight', 'name': '愤怒+战斗'},
    ]

    print("=" * 80)
    print("生物-语言耦合测试")
    print("=" * 80)

    for bio_state in test_cases:
        print(f"\n【{bio_state['name']}】")
        print(f"  输入: {test_text}")
        result = coupler.process(test_text, bio_state)
        print(f"  输出: {result}")

        prosody = coupler.get_prosodic_params(bio_state)
        print(f"  韵律: 语速={prosody['speech_rate']}, "
              f"基频={prosody['pitch_baseline']}Hz, "
              f"嗓音={prosody['voice_register']}, "
              f"清晰度={prosody['articulation_clarity']}")
        print("-" * 60)
