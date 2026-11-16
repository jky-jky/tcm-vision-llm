import paddle
import paddle.nn as nn
from paddle.vision.models import mobilenet_v3_small
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
import gradio as gr
from openai import OpenAI
import os
import httpx
import json
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# 从项目根目录的 .env 加载环境变量
load_dotenv()

# --- 全局配置 ---
MODEL_PATH = 'output/best_model.pdparams'
NUM_CLASSES = 54

# --- 大模型配置（DeepSeek，OpenAI 兼容接口）---
# 具体值统一在项目根目录的 .env 文件中配置，代码里只读取环境变量
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")

if not LLM_API_KEY:
    print("警告：未检测到 LLM_API_KEY，请复制 .env.example 为 .env 并填入你的 API Key。")

MAP_PATH = os.path.join("output", "idx_to_class.json")

# 加载类别映射
try:
    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        RAW_CLASS_MAP = json.load(f)
        CLASS_MAP = {int(k): v for k, v in RAW_CLASS_MAP.items()}
    print(f"成功加载类别映射：{len(CLASS_MAP)} 个类别")
except Exception as e:
    print(f"致命错误: 无法加载类别映射文件 {MAP_PATH} ({e})。请手动创建该文件！")
    CLASS_MAP = {i: f"未知类别_{i}" for i in range(NUM_CLASSES)}


# --- 模型加载 ---
class HerbalModel(nn.Layer):
    def __init__(self, num_classes):
        super(HerbalModel, self).__init__()
        self.backbone = mobilenet_v3_small(pretrained=False)
        last_in_features = self.backbone.classifier[-1].weight.shape[1]
        self.backbone.classifier[-1] = nn.Linear(last_in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_model():
    model = HerbalModel(NUM_CLASSES)
    model_state = paddle.load(MODEL_PATH)
    model.set_state_dict(model_state)
    model.eval()
    return model


try:
    model = load_model()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败，请检查路径和模型结构: {e}")
    model = None

transform = Compose([
    Resize((224, 224)),
    Transpose(),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
])


# --- 大模型客户端 ---
def get_llm_client(token=None):
    """构造一个兼容 OpenAI SDK 的大模型客户端（当前接入 DeepSeek）。"""
    api_key = token if token else LLM_API_KEY
    if not api_key:
        raise RuntimeError("未配置 API Key：请在项目根目录的 .env 文件中设置 LLM_API_KEY")

    proxies = {}
    if os.environ.get("HTTP_PROXY"):
        proxies["http://"] = os.environ.get("HTTP_PROXY")
    if os.environ.get("HTTPS_PROXY"):
        proxies["https://"] = os.environ.get("HTTPS_PROXY")

    client_kwargs = {
        "api_key": api_key,
        "base_url": LLM_BASE_URL,
    }
    if proxies:
        client_kwargs["http_client"] = httpx.Client(proxies=proxies)

    return OpenAI(**client_kwargs)


# --- 核心功能 ---
def predict_herb(image, top_k=5):
    if model is None:
        return {"错误": "模型未加载"}, ""

    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        img_tensor = transform(image)
    except Exception as e:
        return {"错误": f"图像预处理失败: {e}"}, ""

    img_tensor = paddle.to_tensor(img_tensor).unsqueeze(0)

    with paddle.no_grad():
        logits = model(img_tensor)
        probs = paddle.nn.functional.softmax(logits)

    top_probs, top_indices = paddle.topk(probs, k=top_k)
    top_probs = top_probs.numpy()[0]
    top_indices = top_indices.numpy()[0]

    results = {}
    best_name = ""
    for i in range(len(top_indices)):
        idx = int(top_indices[i])
        name = CLASS_MAP.get(idx, f"未知类别_{idx}")
        prob = float(top_probs[i])
        results[name] = prob
        if i == 0 and name:
            best_name = name.split(" ")[0].split("(")[0].strip()

    return results, best_name


def analyze_pharmacology(herb_name, token):
    if not herb_name or "未知类别" in herb_name or "错误" in herb_name:
        return "请先进行图片识别，或者识别结果不确定，无法进行药性解析。"

    try:
        client = get_llm_client(token)
    except Exception as e:
        return f"大模型客户端初始化失败: {str(e)}。请检查 API Key 和环境设置。"

    prompt = f"""
    请作为一位资深中医专家，详细解析中草药【{herb_name}】。
    请包含以下内容：
    1. 【性味归经】：
    2. 【功效主治】：
    3. 【现代药理研究】：（简述其化学成分或现代医学作用）
    4. 【使用禁忌】：

    请确保内容科学、严谨。
    """

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': '你是一个精通中草药药理的分析大师。'},
                {'role': 'user', 'content': prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用大模型失败: {str(e)}。请检查您的 API Key 是否有效以及网络连接是否稳定。"


def answer_question(herb_name, question, token):
    if not herb_name or "未知类别" in herb_name or "错误" in herb_name:
        return "请先完成中草药识别，再进行提问。"
    if not question.strip():
        return "请输入您的问题。"

    try:
        client = get_llm_client(token)
    except Exception as e:
        return f"客户端初始化失败: {str(e)}"

    full_prompt = f"当前讨论的中草药是【{herb_name}】。请回答以下问题：{question}"

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': '你是中医药领域的专业助手，请基于中医理论和现代研究回答问题，语言简洁准确。'},
                {'role': 'user', 'content': full_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"问答失败: {str(e)}"


# --- Gradio 界面（优化布局）---
def pipeline(image, token_input):
    pred_result, best_name = predict_herb(image)
    analysis_result = analyze_pharmacology(best_name, token_input)
    return pred_result, analysis_result, best_name


with gr.Blocks(title="智能中草药检测与药性解析系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🌿 智能中草药检测与药性解析系统</h1>
            <p>基于 MobileNetV3 + DeepSeek 大模型 | 支持识别、解析与交互问答</p>
        </div>
        """
    )

    current_herb_state = gr.State("")

    # === 第一阶段：上传与识别 ===
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📷 上传中草药图片")
            img_input = gr.Image(type="pil", label="支持 JPG/PNG，建议清晰正面图", height=300)
            run_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown("### 📊 识别结果（Top-5）")
            label_output = gr.Label(num_top_classes=5)

    # === 第二阶段：药性解析（识别后显示）===
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📜 药性解析报告")
            text_output = gr.Markdown(elem_classes=["scrollable-box"])

    # === 第三阶段：知识问答（始终可见，但依赖识别结果）===
    gr.Markdown("### ❓ 中草药知识问答")
    with gr.Row():
        with gr.Column():
            gr.Markdown("点击常见问题快速获取答案：")
            with gr.Row():
                btn1 = gr.Button("哪些情况不宜使用？")
                btn2 = gr.Button("有哪些常见配伍？")
                btn3 = gr.Button("有毒性吗？")

    with gr.Row():
        with gr.Column():
            q_input = gr.Textbox(label="自定义问题", placeholder="例如：和黄芪一起用有什么效果？")
            q_btn = gr.Button("💬 提问", variant="secondary")
            q_output = gr.Markdown(elem_classes=["scrollable-box"])

    # CSS 样式：限制高度并支持滚动
    demo.css = """
    .scrollable-box {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #fafafa;
    }
    """

    # 绑定事件
    run_btn.click(
        fn=pipeline,
        inputs=[img_input],
        outputs=[label_output, text_output, current_herb_state]
    )

    # 预设问题
    for btn, q in zip([btn1, btn2, btn3], [
        "哪些情况不宜使用？",
        "有哪些常见配伍？",
        "有毒性吗？"
    ]):
        btn.click(
            fn=lambda h, q=q, t="": answer_question(h, q, t),
            inputs=[current_herb_state],
            outputs=q_output
        )

    # 自由提问
    q_btn.click(
        fn=answer_question,
        inputs=[current_herb_state, q_input],
        outputs=q_output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)