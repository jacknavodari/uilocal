import os
import re
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple
import base64
import requests
import json

import gradio as gr
from huggingface_hub import InferenceClient

import modelscope_studio.components.base as ms
import modelscope_studio.components.legacy as legacy
import modelscope_studio.components.antd as antd

# Configuration
SystemPrompt = """You are a helpful coding assistant. You help users create applications by generating code based on their requirements. 
When asked to create an application, you should:
1. Understand the user's requirements
2. Generate clean, working code
3. Provide HTML output when appropriate for web applications
4. Include necessary comments and documentation
5. Ensure the code is functional and follows best practices

If an image is provided, analyze it and use the visual information to better understand the user's requirements.

Always respond with code that can be executed or rendered directly.

Always output only the HTML code inside a ```html ... ``` code block, and do not include any explanations or extra text."""

# Add Ollama API endpoint
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')

def get_ollama_models():
    """Fetch available Ollama models from the local Ollama server."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        response.raise_for_status()
        data = response.json()
        models = data.get('models', [])
        return [
            {
                "name": m.get('name', m.get('model', 'Unknown')),
                "id": m.get('name', m.get('model', 'Unknown')),
                "description": f"Ollama model: {m.get('name', m.get('model', 'Unknown'))}"
            }
            for m in models
        ]
    except Exception as e:
        # Fallback to default model if Ollama is not running
        return [
            {
                "name": "Ollama (Local)",
                "id": "ollama",
                "description": "Local Ollama model for code generation (Ollama not running or no models found)"
            }
        ]

# Available models
AVAILABLE_MODELS = get_ollama_models()

DEMO_LIST = [
    {
        "title": "Todo App",
        "description": "Create a simple todo application with add, delete, and mark as complete functionality"
    },
    {
        "title": "Calculator",
        "description": "Build a basic calculator with addition, subtraction, multiplication, and division"
    },
    {
        "title": "Weather Dashboard",
        "description": "Create a weather dashboard that displays current weather information"
    },
    {
        "title": "Chat Interface",
        "description": "Build a chat interface with message history and user input"
    },
    {
        "title": "E-commerce Product Card",
        "description": "Create a product card component for an e-commerce website"
    },
    {
        "title": "Login Form",
        "description": "Build a responsive login form with validation"
    },
    {
        "title": "Dashboard Layout",
        "description": "Create a dashboard layout with sidebar navigation and main content area"
    },
    {
        "title": "Data Table",
        "description": "Build a data table with sorting and filtering capabilities"
    },
    {
        "title": "Image Gallery",
        "description": "Create an image gallery with lightbox functionality and responsive grid layout"
    },
    {
        "title": "UI from Image",
        "description": "Upload an image of a UI design and I'll generate the HTML/CSS code for it"
    }
]

# HF Inference Client
# Create InferenceClient instance
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['API_PROXY'] = 'http://api.wlai.vip'

def query_huggingface_api(input_text):
    api_url = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
    }
    
    # Add timeout parameter to the request
    try:
        response = requests.post(
            api_url, 
            json={"inputs": input_text}, 
            headers=headers,
            timeout=30  # 30 seconds timeout
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

client = None  # Disable the InferenceClient for now

def ollama_generate(prompt: str, system_prompt: str = SystemPrompt, model: str = "phi4:14b"):
    """Generate response using local Ollama model"""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True
    }
    
    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            content = ""
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    try:
                        json_data = json.loads(data)
                        if 'response' in json_data:
                            chunk = json_data['response']
                            content += chunk
                            yield {
                                "content": content,
                                "is_error": False
                            }
                    except json.JSONDecodeError:
                        continue
            
    except requests.exceptions.RequestException as e:
        yield {
            "content": f"Error connecting to Ollama: {str(e)}",
            "is_error": True
        }

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': 'system', 'content': system}]
    for h in history:
        # Handle multimodal content in history
        user_content = h[0]
        if isinstance(user_content, list):
            # Extract text from multimodal content
            text_content = ""
            for item in user_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
            user_content = text_content if text_content else str(user_content)
        
        messages.append({'role': 'user', 'content': user_content})
        messages.append({'role': 'assistant', 'content': h[1]})
    return messages

def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == 'system'
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        # Extract text content from multimodal messages for history
        user_content = q['content']
        if isinstance(user_content, list):
            text_content = ""
            for item in user_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
            user_content = text_content if text_content else str(user_content)
        
        history.append([user_content, r['content']])
    return history

def remove_code_block(text):
    # Try to match code blocks with language markers
    patterns = [
        r'```(?:html|HTML)\n([\s\S]+?)\n```',  # Match ```html or ```HTML
        r'```\n([\s\S]+?)\n```',               # Match code blocks without language markers
        r'```([\s\S]+?)```'                      # Match code blocks without line breaks
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            return extracted
    # If no code block is found, check if the entire text is HTML
    if text.strip().startswith('<!DOCTYPE html>') or text.strip().startswith('<html'):
        return text.strip()
    return text.strip()

def history_render(history: History):
    return gr.update(open=True), history

def clear_history():
    return []

def update_image_input_visibility(model):
    """Update image input visibility based on selected model"""
    is_ernie_vl = model.get("id") == "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT"
    return gr.update(visible=is_ernie_vl)

def process_image_for_model(image):
    """Convert image to base64 for model input"""
    if image is None:
        return None
    
    # Convert numpy array to PIL Image if needed
    import io
    import base64
    import numpy as np
    from PIL import Image
    
    # Handle numpy array from Gradio
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_multimodal_message(text, image=None):
    """Create a multimodal message with text and optional image"""
    if image is None:
        return {"role": "user", "content": text}
    
    content = [
        {
            "type": "text",
            "text": text
        },
        {
            "type": "image_url",
            "image_url": {
                "url": process_image_for_model(image)
            }
        }
    ]
    
    return {"role": "user", "content": content}

def send_to_sandbox(code):
    # Add a wrapper to inject necessary permissions and ensure full HTML
    wrapped_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <script>
            // Safe localStorage polyfill
            const safeStorage = {{
                _data: {{}},
                getItem: function(key) {{ return this._data[key] || null; }},
                setItem: function(key, value) {{ this._data[key] = value; }},
                removeItem: function(key) {{ delete this._data[key]; }},
                clear: function() {{ this._data = {{}}; }}
            }};
            Object.defineProperty(window, 'localStorage', {{
                value: safeStorage,
                writable: false
            }});
            window.onerror = function(message, source, lineno, colno, error) {{
                console.error('Error:', message);
            }};
        </script>
    </head>
    <body>
        {code}
    </body>
    </html>
    """
    encoded_html = base64.b64encode(wrapped_code.encode('utf-8')).decode('utf-8')
    data_uri = f"data:text/html;charset=utf-8;base64,{encoded_html}"
    iframe = f'<iframe src="{data_uri}" width="100%" height="920px" sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-presentation" allow="display-capture"></iframe>'
    return iframe

def demo_card_click(e: gr.EventData):
    try:
        # Get the index from the event data
        if hasattr(e, '_data') and e._data:
            # Try different ways to get the index
            if 'index' in e._data:
                index = e._data['index']
            elif 'component' in e._data and 'index' in e._data['component']:
                index = e._data['component']['index']
            elif 'target' in e._data and 'index' in e._data['target']:
                index = e._data['target']['index']
            else:
                # If we can't get the index, try to extract it from the card data
                index = 0
        else:
            index = 0
        
        # Ensure index is within bounds
        if index >= len(DEMO_LIST):
            index = 0
            
        return DEMO_LIST[index]['description']
    except (KeyError, IndexError, AttributeError) as e:
        # Return the first demo description as fallback
        return DEMO_LIST[0]['description']

# Main application
with gr.Blocks(css_paths="app.css") as demo:
    history = gr.State([])
    setting = gr.State({
        "system": SystemPrompt,
    })
    current_model = gr.State(AVAILABLE_MODELS[0])  # Default to Ollama (only available model)

    with ms.Application() as app:
        with antd.ConfigProvider():
            with antd.Row(gutter=[32, 12]) as layout:
                with antd.Col(span=24, md=8):
                    with antd.Flex(vertical=True, gap="middle", wrap=True):
                        header = gr.HTML("""
                                  <div class="left_header">
                                   <img src="https://huggingface.co/spaces/akhaliq/anycoder/resolve/main/Animated_Logo_Video_Ready.gif" width="200px" />
                                   <h1>uilocal</h1>
                                  </div>
                                   """)
                        current_model_display = gr.Markdown("**Current Model:** DeepSeek R1")
                        input = antd.InputTextarea(
                            size="large", allow_clear=True, placeholder="Please enter what kind of application you want")
                        image_input = gr.Image(label="Upload an image (only for ERNIE-4.5-VL model)", visible=False)
                        btn = antd.Button("send", type="primary", size="large")
                        clear_btn = antd.Button("clear history", type="default", size="large")

                        antd.Divider("examples")
                        with antd.Flex(gap="small", wrap=True) as examples_flex:
                            for i, demo_item in enumerate(DEMO_LIST):
                                with antd.Card(hoverable=True, title=demo_item["title"]) as demoCard:
                                    antd.CardMeta(description=demo_item["description"])
                                demoCard.click(lambda e, idx=i: (DEMO_LIST[idx]['description'], None), outputs=[input, image_input])

                        antd.Divider("setting")
                        with antd.Flex(gap="small", wrap=True) as setting_flex:
                            settingPromptBtn = antd.Button(
                                "⚙️ set system Prompt", type="default")
                            modelBtn = antd.Button("🤖 switch model", type="default")
                            codeBtn = antd.Button("🧑‍💻 view code", type="default")
                            historyBtn = antd.Button("📜 history", type="default")

                    with antd.Modal(open=False, title="set system Prompt", width="800px") as system_prompt_modal:
                        systemPromptInput = antd.InputTextarea(
                            SystemPrompt, auto_size=True)

                    settingPromptBtn.click(lambda: gr.update(
                        open=True), inputs=[], outputs=[system_prompt_modal])
                    system_prompt_modal.ok(lambda input: ({"system": input}, gr.update(
                        open=False)), inputs=[systemPromptInput], outputs=[setting, system_prompt_modal])
                    system_prompt_modal.cancel(lambda: gr.update(
                        open=False), outputs=[system_prompt_modal])

                    with antd.Modal(open=False, title="Select Model", width="600px") as model_modal:
                        with antd.Flex(vertical=True, gap="middle"):
                            for i, model in enumerate(AVAILABLE_MODELS):
                                with antd.Card(hoverable=True, title=model["name"]) as modelCard:
                                    antd.CardMeta(description=model["description"])
                                modelCard.click(lambda m=model: (m, gr.update(open=False), f"**Current Model:** {m['name']}", update_image_input_visibility(m)), outputs=[current_model, model_modal, current_model_display, image_input])

                    modelBtn.click(lambda: gr.update(open=True), inputs=[], outputs=[model_modal])

                    with antd.Drawer(open=False, title="code", placement="left", width="750px") as code_drawer:
                        code_output = legacy.Markdown()

                    codeBtn.click(lambda: gr.update(open=True),
                                  inputs=[], outputs=[code_drawer])
                    code_drawer.close(lambda: gr.update(
                        open=False), inputs=[], outputs=[code_drawer])

                    with antd.Drawer(open=False, title="history", placement="left", width="900px") as history_drawer:
                        history_output = legacy.Chatbot(show_label=False, flushing=False, height=960, elem_classes="history_chatbot")

                    historyBtn.click(history_render, inputs=[history], outputs=[history_drawer, history_output])
                    history_drawer.close(lambda: gr.update(
                        open=False), inputs=[], outputs=[history_drawer])

                with antd.Col(span=24, md=16):
                    with ms.Div(elem_classes="right_panel"):
                        gr.HTML('<div class="render_header"><span class="header_btn"></span><span class="header_btn"></span><span class="header_btn"></span></div>')
                        # Move sandbox outside of tabs for always-on visibility
                        sandbox = gr.HTML(elem_classes="html_content")
                        with antd.Tabs(active_key="empty", render_tab_bar="() => null") as state_tab:
                            with antd.Tabs.Item(key="empty"):
                                empty = antd.Empty(description="empty input", elem_classes="right_content")
                            with antd.Tabs.Item(key="loading"):
                                loading = antd.Spin(True, tip="coding...", size="large", elem_classes="right_content")

            def generation_code(query: Optional[str], image: Optional[gr.Image], _setting: Dict[str, str], _history: Optional[History], _current_model: Dict):
                if query is None:
                    query = ''
                if _history is None:
                    _history = []
                messages = history_to_messages(_history, _setting['system'])
                
                # Create multimodal message if image is provided
                if image is not None:
                    messages.append(create_multimodal_message(query, image))
                else:
                    messages.append({'role': 'user', 'content': query})

                try:
                    if _current_model["id"] in [m["id"] for m in AVAILABLE_MODELS]:
                        # Use selected Ollama model for code generation
                        content = ""
                        for chunk in ollama_generate(query, _setting['system'], model=_current_model["id"]):
                            content = chunk["content"]
                            yield {
                                code_output: content,
                                state_tab: gr.update(active_key="loading"),
                                code_drawer: gr.update(open=True),
                            }
                        # Final response
                        _history = messages_to_history(messages + [{
                            'role': 'assistant',
                            'content': content
                        }])
                        yield {
                            code_output: content,
                            history: _history,
                            sandbox: send_to_sandbox(remove_code_block(content)),
                            state_tab: gr.update(active_key="render"),
                            code_drawer: gr.update(open=False),
                        }
                    else:
                        # Existing HuggingFace Hub logic for other models
                        completion = client.chat.completions.create(
                            model=_current_model["id"],
                            messages=messages,
                            stream=True,
                            max_tokens=5000  # Higher max_tokens for more complete applications while maintaining reasonable speed
                        )
                        
                        content = ""
                        for chunk in completion:
                            if chunk.choices[0].delta.content:
                                content += chunk.choices[0].delta.content
                                yield {
                                    code_output: content,
                                    state_tab: gr.update(active_key="loading"),
                                    code_drawer: gr.update(open=True),
                                }
                        
                        # Final response
                        _history = messages_to_history(messages + [{
                            'role': 'assistant',
                            'content': content
                        }])
                        
                        yield {
                            code_output: content,
                            history: _history,
                            sandbox: send_to_sandbox(remove_code_block(content)),
                            state_tab: gr.update(active_key="render"),
                            code_drawer: gr.update(open=False),
                        }
                        
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    yield {
                        code_output: error_message,
                        state_tab: gr.update(active_key="empty"),
                        code_drawer: gr.update(open=True),
                    }

            btn.click(
                generation_code,
                inputs=[input, image_input, setting, history, current_model],
                outputs=[code_output, history, sandbox, state_tab, code_drawer]
            )
            
            clear_btn.click(clear_history, inputs=[], outputs=[history])



if __name__ == "__main__":
    demo.queue(default_concurrency_limit=20).launch(ssr_mode=False)