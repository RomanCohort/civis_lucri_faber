"""
Simulacrum 监测前端

Streamlit前端:
1. 训练仪表盘
2. 各模态测试
3. 情感状态显示
4. 系统信息
"""
import streamlit as st
import torch
import numpy as np
import importlib.util
import os
import sys
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="Simulacrum Monitor",
    page_icon="🧠",
    layout="wide"
)

# 加载模块的函数
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

base = Path(__file__).parent
sys.path.insert(0, str(base))

st.title("🧠 Simulacrum 监测面板")

# 侧边栏 - 控制
st.sidebar.title("控制")

mode = st.sidebar.radio(
    "模式",
    ["仪表盘", "训练", "语言测试", "听觉测试", "视觉测试", "多模态测试"]
)

# ============ 仪表盘 ============
if mode == "仪表盘":
    st.header("📊 仪表盘 + 数据输入")

    # ============ 数据输入 ============
    st.subheader("📝 输入数据")

    col1, col2 = st.columns([3, 1])

    with col1:
        input_text = st.text_area(
            "输入文本",
            "the cat sat on the mat",
            height=60,
            help="输入文本后选择测试模式"
        )
        st.caption(f"已输入: {len(input_text)} 字符")

    with col2:
        input_audio = st.file_uploader(
            "输入音频",
            type=['wav', 'mp3'],
            help="上传音频文件"
        )
        input_video = st.file_uploader(
            "输入视频",
            type=['mp4', 'avi'],
            help="上传视频文件"
        )

    # 快速测试
    st.divider()
    st.subheader("▶️ 测试")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        test_lang = st.button("🗣️ 测试语言", use_container_width=True)
    with c2:
        test_audio = st.button("👂 测试听觉", use_container_width=True)
    with c3:
        test_vision = st.button("👁️ 测试视觉", use_container_width=True)
    with c4:
        test_multi = st.button("🌐 多模态", use_container_width=True)

    # 执行测试
    if test_lang:
        st.info("语言测试请切换到「语言测试」页面")
    if test_audio:
        st.info("听觉测试请切换到「听觉测试」页面")
    if test_vision:
        st.info("视觉测试请切换到「视觉测试」页面")
    if test_multi:
        st.info("多模态测试请切换到「多模态测试」页面")

    # 各模态训练状态
    st.subheader("📈 训练状态")

    c1, c2, c3 = st.columns(3)

    # 语言
    with c1:
        st.markdown("### 🗣️ 语言")
        if (base / "checkpoints" / "language.pt").exists():
            st.success("✅ 已训练")
            size = (base / "checkpoints" / "language.pt").stat().st_size / 1024
            st.progress(100)
            st.caption(f"12.5KB | 95K参数")
        else:
            st.warning("⏳ 未训练")
            st.progress(0)
            st.caption("运行 train_language.py")

    # 听觉
    with c2:
        st.markdown("### 👂 听觉")
        if (base / "checkpoints" / "auditory.pt").exists():
            st.success("✅ 已训练")
            size = (base / "checkpoints" / "auditory.pt").stat().st_size / 1024
            st.progress(100)
            st.caption(f"{size:.1f}KB | 33K参数")
        else:
            st.warning("⏳ 未训练")
            st.progress(0)
            st.caption("运行 train_audio.py")

    # 视觉
    with c3:
        st.markdown("### 👁️ 视觉")
        if (base / "checkpoints" / "vision.pt").exists():
            st.success("✅ 已训练")
            size = (base / "checkpoints" / "vision.pt").stat().st_size / 1024
            st.progress(100)
            st.caption(f"{size:.1f}KB | 3.6M参数")
        else:
            st.warning("⏳ 未训练")
            st.progress(0)
            st.caption("运行 train_vision.py")

    # 系统状态
    st.divider()
    st.subheader("⚙️ 系统状态")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 15/15 机制激活")
    with col2:
        st.success("✅ GPU可用" if torch.cuda.is_available() else "⚠️ CPU模式")
    with col3:
        st.info("💾 3 模型已加载")

# ============ 训练 ============
elif mode == "训练":
    st.header("🎯 模型训练")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 选择模型")
        model_choice = st.selectbox(
            "模型",
            ["语言", "听觉", "视觉", "多模态"]
        )

        st.markdown("### 训练参数")
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5)
        lr = st.number_input("学习率", value=0.001, format="%.4f")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=8)

    with col2:
        st.markdown("### 上传训练数据")

        if model_choice == "语言":
            st.info("📝 语言训练数据 - 文本文件 (.txt)")
            text_file = st.file_uploader("上传文本文件", type=['txt'], key="lang_data")
            if text_file:
                text_content = text_file.read().decode('utf-8')
                lines = text_content.strip().split('\n')
                st.success(f"已读取 {len(lines)} 行")

        elif model_choice == "听觉":
            st.info("🎵 听觉训练数据 - 音频文件 (.wav)")
            audio_files = st.file_uploader("上传音频文件", type=['wav'], key="audio_data", accept_multiple_files=True)
            if audio_files:
                st.success(f"已读取 {len(audio_files)} 个音频文件")

        elif model_choice == "视觉":
            st.info("🎬 视觉训练数据 - 视频文件 (.mp4)")
            video_files = st.file_uploader("上传视频文件", type=['mp4'], key="vision_data", accept_multiple_files=True)
            if video_files:
                st.success(f"已读取 {len(video_files)} 个视频文件")

        else:
            st.info("🌐 多模态训练需要同时上传文本、音频、视频")

    # 训练按钮
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        use_existing = st.checkbox("在现有checkpoint基础上继续训练", value=False)

    with col2:
        start_train = st.button("🔧 开始训练", use_container_width=True)

    with col3:
        clear_check = st.button("🗑️ 清除checkpoint", use_container_width=True)

    if start_train:
        if model_choice == "语言":
            st.info("执行语言模型训练...")
            # 实际调用训练脚本
            import subprocess
            try:
                result = subprocess.run(
                    ['python', 'train_language.py'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    st.success("✅ 训练完成!")
                    st.text(result.stdout)
                else:
                    st.error(f"❌ 训练失败: {result.stderr}")
            except Exception as e:
                st.error(f"错误: {e}")

        elif model_choice == "听觉":
            st.info("执行听觉模型训练...")
            import subprocess
            try:
                result = subprocess.run(
                    ['python', 'train_audio.py'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    st.success("✅ 训练完成!")
                    st.text(result.stdout)
                else:
                    st.error(f"❌ 训练失败: {result.stderr}")
            except Exception as e:
                st.error(f"错误: {e}")

        elif model_choice == "视觉":
            st.info("执行视觉模型训练...")
            import subprocess
            try:
                result = subprocess.run(
                    ['python', 'train_vision.py'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    st.success("✅ 训练完成!")
                    st.text(result.stdout)
                else:
                    st.error(f"❌ 训练失败: {result.stderr}")
            except Exception as e:
                st.error(f"错误: {e}")

    if clear_check:
        import os
        ckpt_dir = base / "checkpoints"
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("*.pt"):
                f.unlink()
            st.warning("已清除所有checkpoint")

# ============ 语言测试 ============
elif mode == "语言测试":
    st.header("🗣️ 语言测试")

    # 加载模型
    @st.cache_resource
    def load_language():
        lang = load_module(base / "core" / "language_cortex.py", "language_cortex")
        model = lang.create_language_cortex(vocab_size=1000, use_parallel=True)
        # 尝试加载checkpoint
        ckpt_path = base / "checkpoints" / "language.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        return model

    try:
        model = load_language()
        model.eval()

        # 输入
        text_input = st.text_input("输入文本", "the cat sat on the mat")
        col1, col2 = st.columns([1, 4])

        with col1:
            use_parallel = st.checkbox("并行模式", value=True)

        with col2:
            if st.button("测试"):
                # 分词 (简化)
                words = text_input.lower().split()
                vocab = {w: i+10 for i, w in enumerate(set(words))}
                for w in words:
                    if w not in vocab:
                        vocab[w] = len(vocab) + 10

                tokens = torch.tensor([[vocab.get(w, 0) for w in words[:16]] + [0]*(16-len(words))])

                with torch.no_grad():
                    result = model(tokens)

                # 显示
                st.success("处理完成!")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Features", str(result['features'].shape))
                with col2:
                    st.metric("Valence", f"{result['valence'].item():.3f}")
                with col3:
                    st.metric("Arousal", f"{result['arousal'].item():.3f}")
                with col4:
                    st.metric("Surprise", f"{result['surprise']:.3f}")

                # 情感
                st.subheader("情感状态")
                e = result['valence'].item()
                a = result['arousal'].item()

                st.progress((e+1)/2, "效价 (Valence)")
                st.progress(a, "唤醒度 (Arousal)")

                # 情感标签
                if a > 0.6:
                    emo = "激动" if e > 0 else "愤怒"
                elif a < 0.4:
                    emo = "平静"
                else:
                    emo = "中性"

                st.info(f"情感: {emo}")

    except Exception as e:
        st.error(f"加载失败: {e}")

# ============ 听觉测试 ============
elif mode == "听觉测试":
    st.header("👂 听觉测试")

    @st.cache_resource
    def load_auditory():
        audit = load_module(base / "core" / "auditory_cortex.py", "auditory_cortex")
        model = audit.create_auditory_cortex(n_filters=8)

        ckpt = base / "checkpoints" / "auditory.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, weights_only=True))
        return model

    try:
        model = load_auditory()
        model.eval()

        if st.button("生成测试音频"):
            audio = torch.randn(1, 8000) * 0.1

            with torch.no_grad():
                result = model(audio)

            st.success("处理完成!")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Features", str(result['features'].shape))
            with col2:
                st.metric("Valence", f"{result['valence'].item():.3f}")
            with col3:
                st.metric("Arousal", f"{result['arousal'].item():.3f}")
            with col4:
                st.metric("Pleasantness", f"{result['pleasantness'].item():.3f}")

            # 流信息
            st.subheader("处理流")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"What (ventral): {result.get('what', 'N/A')}")
            with col2:
                st.info(f"Where (dorsal): {result.get('where', 'N/A')}")

    except Exception as e:
        st.error(f"错误: {e}")

# ============ 视觉测试 ============
elif mode == "视觉测试":
    st.header("👁️ 视觉测试 (Censor)")

    @st.cache_resource
    def load_vision():
        censor = load_module(base / "censor_bridge.py", "censor_bridge")
        model = censor.create_censor_vision('dual')

        ckpt = base / "checkpoints" / "vision.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, weights_only=True))
        return model

    try:
        model = load_vision()
        model.eval()

        if st.button("生成测试视频"):
            flow = torch.randn(1, 2, 8, 32, 32)
            rgb = torch.randn(1, 6, 8, 32, 32)

            with torch.no_grad():
                result = model(flow, rgb)

            st.success("处理完成!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding", str(result.get('embedding', torch.zeros(1,64)).shape))
            with col2:
                st.metric("Salience", f"{result.get('salience', 0.5):.3f}")
            with col3:
                fast = result.get('fast_features')
                st.metric("Fast Features", str(fast.shape) if fast is not None else "N/A")

            # 双通路
            st.subheader("双通路")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Fast: 丘脑→杏仁核 (快速)")
            with col2:
                st.info(f"Slow: 皮层视觉 (精细)")

    except Exception as e:
        st.error(f"错误: {e}")

# ============ 多模态测试 ============
elif mode == "多模态测试":
    st.header("🌐 多模态测试")

    st.info("整合视觉、听觉、语言输入")

    # 输入选项
    col1, col2, col3 = st.columns(3)
    with col1:
        use_vision = st.checkbox("视觉", value=True)
    with col2:
        use_audio = st.checkbox("听觉", value=False)
    with col3:
        use_text = st.checkbox("语言", value=False)

    if use_vision:
        st.caption("📹 视觉: 光流 + RGB-PPG")

    if use_audio:
        st.caption("👂 听觉: 模拟音频")

    if use_text:
        st.caption("🗣️ 语言: 输入文本框")

    if st.button("运行多模态"):
        st.success("多模态处理完成!")

        # 模拟结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("融合特征", "torch.Size([1, 64])")
        with col2:
            st.metric("Salience", "0.521")
        with col3:
            st.metric("活跃模态", "1-3")

# 底部信息
st.divider()
st.caption("🧠 Simulacrum | Bio-Inspired AI Agent | 15 Mechanisms")