"""Simulacrum Flask Backend - 为 Electron 前端提供 API

启动: python electron_app/server.py [--provider mock] [--port 5200]
"""
import sys
import os
import json
import time
import threading
import collections
import argparse
import random

# 项目根目录的父目录 (simulacrum 包所在的目录)
# server.py 在 electron_app/ 下
# D:\simulacrum\electron_app\server.py → 需要 D:\ 在 sys.path
# 才能 import simulacrum
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)        # D:\simulacrum
_parent_dir = os.path.dirname(_project_root)      # D:\
for _p in [_parent_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import Flask, request, jsonify, Response, send_from_directory

# ── 初始化 Simulacrum Agent ──────────────────────────────────────────

_agent = None
_history = collections.deque(maxlen=500)
_state_history = collections.deque(maxlen=120)   # 最近 120 步
_step_counter = 0
_lock = threading.Lock()

app = Flask(__name__, static_folder="renderer", static_url_path="")


def _init_agent(provider="mock"):
    global _agent
    from simulacrum.utils.config import load_config
    from simulacrum.core.agent import Simulacrum

    config = load_config(
        llm_provider=provider,
        device="cpu",
        initial_balance=100.0,
    )
    _agent = Simulacrum(config=config)
    print(f"[OK] Simulacrum Agent initialized (provider={provider})")


# ── API 路由 ──────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("renderer", "index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """发送消息给 agent 并返回回复"""
    data = request.get_json(force=True)
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    global _step_counter
    try:
        response = _agent.chat(user_text)
        with _lock:
            _step_counter += 1
            entry = {
                "id": _step_counter,
                "time": time.time(),
                "user": user_text,
                "text": response.text,
                "emotion": response.emotion,
                "arousal": response.arousal,
                "valence": response.valence,
                "llm_params": response.llm_params or {},
                "cognitive_gate": response.cognitive_gate or {},
                "quality_filter": response.quality_filter or {},
                "learning_active": response.learning_active,
                "tool_calls": response.tool_calls or [],
            }
            _history.append(entry)
            # 快照内部状态
            state_snapshot = _snapshot_state()
            state_snapshot["step"] = _step_counter
            _state_history.append(state_snapshot)

        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/state")
def api_state():
    """返回当前 agent 内部状态 (最新快照)"""
    state = _snapshot_state()
    return jsonify(state)


@app.route("/api/state/stream")
def api_state_stream():
    """SSE: 实时推送内部状态 (每 300ms)"""
    def generate():
        while True:
            state = _snapshot_state()
            yield f"data: {json.dumps(state)}\n\n"
            time.sleep(0.3)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/state_history")
def api_state_history():
    """返回历史状态数组 (用于前端图表)"""
    with _lock:
        data = list(_state_history)
    return jsonify(data)


@app.route("/api/history")
def api_history():
    """返回聊天历史"""
    with _lock:
        data = list(_history)
    return jsonify(data)


@app.route("/api/pharma", methods=["POST"])
def api_pharma():
    """药理学操作接口"""
    data = request.get_json(force=True)
    action = data.get("action")
    target = data.get("target")
    value = data.get("value")

    pharma = getattr(_agent, 'pharma', None)
    if not pharma:
        return jsonify({"error": "Pharmacology not available"}), 400

    try:
        result = None
        if action == "inject":
            result = pharma.inject(target, float(value) if value else 0.5)
        elif action == "reduce":
            result = pharma.reduce(target, float(value) if value else 0.5)
        elif action == "anesthetize":
            result = pharma.anesthetize(target)
        elif action == "activate":
            result = pharma.activate(target)
        elif action == "lesion":
            result = pharma.lesion(target)
        elif action == "prescribe":
            result = pharma.prescribe(target)
        elif action == "reset":
            pharma.reset()
            result = {"ok": True, "message": "Reset to snapshot"}
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400

        state = _snapshot_state()
        return jsonify({"result": result, "state": state})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pharma/presets")
def api_pharma_presets():
    """获取药理预设列表"""
    from simulacrum.core.neuro_pharmacology import _PRESETS
    presets = {}
    for k, v in _PRESETS.items():
        presets[k] = {
            "name": v.get("name", k),
            "effects": v.get("effects", []),
            "description": v.get("description", ""),
        }
    return jsonify(presets)


@app.route("/api/pharma/regions")
def api_pharma_regions():
    """获取脑区列表"""
    from simulacrum.core.neuro_pharmacology import Region
    return jsonify([r.value for r in Region])


@app.route("/api/pharma/neurotransmitters")
def api_pharma_neurotransmitters():
    """获取神经递质列表"""
    from simulacrum.core.neuro_pharmacology import Neurotransmitter
    return jsonify([n.value for n in Neurotransmitter])


# ── 辅助函数 ──────────────────────────────────────────────────

def _snapshot_state():
    """从 agent._internal_state 提取纯 dict 快照"""
    if not _agent:
        return {"error": "No agent"}
    state = {}
    for k, v in _agent._internal_state.items():
        if isinstance(v, (int, float)):
            state[k] = round(v, 4)
        elif isinstance(v, bool):
            state[k] = v
        elif isinstance(v, str):
            state[k] = v
        elif isinstance(v, list):
            state[k] = [round(x, 3) if isinstance(x, float) else x for x in v[:20]]
        else:
            state[k] = str(v)[:60] if v is not None else None
    return state


# ── 入口 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulacrum Backend Server")
    parser.add_argument("--provider", default="mock", help="LLM provider")
    parser.add_argument("--port", type=int, default=5200, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("  Simulacrum Neuro Engine - Backend Server")
    print("=" * 50)
    _init_agent(args.provider)
    print(f"[SERVE] http://{args.host}:{args.port}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
