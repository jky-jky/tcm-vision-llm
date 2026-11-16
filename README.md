# 🌿 智能中草药检测与药性解析系统

基于 **MobileNetV3 + DeepSeek 大模型** 的中草药识别与知识问答系统。上传一张中草药图片，系统即可完成图像识别，并调用大模型生成结构化的药性解析，还支持围绕该药材的交互问答。

## ✨ 功能特性

- **图片识别**：基于 PaddlePaddle 的 MobileNetV3-small 对 54 类中草药进行分类，输出 Top-5 概率。
- **药性解析**：调用 DeepSeek（OpenAI 兼容接口）生成【性味归经 / 功效主治 / 现代药理研究 / 使用禁忌】的结构化报告。
- **知识问答**：内置常见问题（禁忌、配伍、毒性）一键提问，也支持自由提问。
- **隐私安全**：密钥等敏感信息统一放在 `.env`，不写死在代码中，也不会提交到仓库。

## 🧱 技术栈

| 类别 | 技术 |
| --- | --- |
| 深度学习 | PaddlePaddle、MobileNetV3 |
| 界面 | Gradio |
| 大模型 | DeepSeek（OpenAI 兼容 SDK） |
| 配置 | python-dotenv |

## 📁 目录结构

```
中医药识别/
├── APP.py                  # Gradio 主程序：识别 + 药性解析 + 问答
├── train.py                # 模型训练脚本
├── labels_map.py           # 类别标签映射
├── output/
│   ├── best_model.pdparams # 训练好的模型权重（体积大，未纳入仓库）
│   ├── idx_to_class.json   # 类别索引映射（运行时必需）
│   ├── Figure_1.png
│   └── training_metrics.png
├── data/                   # 训练图片数据集（约 10350 张，未纳入仓库）
├── requirements.txt
├── .env.example            # 环境变量模板
└── .gitignore
```

> `data/`、模型权重文件以及课设报告（`.docx/.pptx`）因体积或用途原因未纳入版本管理。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制模板为 `.env`，并填入你自己的 DeepSeek API Key：

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

`.env` 示例：

```dotenv
LLM_API_KEY=your_deepseek_api_key_here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `LLM_API_KEY` | DeepSeek API Key | 必填 |
| `LLM_BASE_URL` | OpenAI 兼容接口地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-flash` |

### 3. 准备模型与数据

- 将训练好的 `best_model.pdparams` 放入 `output/` 目录；
- 确认 `output/idx_to_class.json` 存在（类别映射，运行时必需）。

> 模型权重与数据集因体积较大未包含在仓库中，请自行训练生成或另行下载。

### 4. 运行

```bash
python APP.py
```

启动后按提示在浏览器中打开即可使用。

## ⚠️ 注意事项

- `.env` 包含真实密钥，已被 `.gitignore` 忽略，**请勿提交或公开泄露**。
- 如需将模型权重纳入仓库，可改用 [Git LFS](https://git-lfs.com/)，或从 `.gitignore` 中移除 `output/*.pdparams`。
- 数据集 `data/` 体积约 1.25 GB，不建议提交到 Git。

## 📄 License

本项目仅用于学习与课程设计用途。
