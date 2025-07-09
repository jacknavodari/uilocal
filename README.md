---
title: uilocal
emoji: 🏢
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
disable_embedding: true
hf_oauth: true
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# uilocal

**A local UI code generation assistant powered by Ollama and enhanced with GPT-4.1.**

---

## ✨ Features

- **Ollama Model Selection:**
  - Automatically detects and lists all available Ollama models on your local server.
  - Easily switch between models for code generation.

- **Modern UI Playground:**
  - Built with Gradio and ModelScope Studio components for a beautiful, interactive experience.
  - Example prompts and demo cards for quick testing.

- **Multimodal Input:**
  - Supports text and image input (for compatible models).

- **HTML/CSS Code Generation:**
  - Generates clean, ready-to-use HTML/CSS code for your UI ideas.
  - Output is always in a code block for easy copy-paste or preview.

- **Sandbox Preview:**
  - Instantly preview generated HTML/CSS in a secure sandboxed iframe.

- **History & Code View:**
  - View your previous generations and inspect the raw code.

- **Custom System Prompt:**
  - Easily adjust the system prompt to guide the assistant's behavior.

- **Local-First, Private:**
  - All code generation happens locally via your Ollama models. No cloud required for main features.

---

## 🚀 Getting Started

1. **Install Ollama** and pull your favorite models (e.g. `phi4:14b`, `llama3`, etc).
2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```sh
   python app.py
   ```
4. **Open your browser** to the provided Gradio link.

---

## 🛠️ Requirements
- Python 3.8+
- Ollama running locally (default: `http://localhost:11434`)
- [Gradio](https://gradio.app/), [modelscope-studio](https://pypi.org/project/modelscope-studio/)

---

## 🧠 About
- **uilocal** is a fork/modification of an open-source coding assistant, now enhanced for local Ollama model selection and UI code generation.
- **Enhanced and vibecoded by Novac Ionel using GPT-4.1.**
- For feedback or improvements, open an issue or PR!

---

## ⚡ Credits
- Built with [Gradio](https://gradio.app/), [ModelScope Studio](https://modelscope.cn/), and [Ollama](https://ollama.com/).
- Original inspiration: AnyCoder, HuggingFace Spaces.

---

**Happy coding!**