"""
AI Chatbot Application with Ollama Integration
"""
import os
import json
import requests
import gradio as gr

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
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'phi4:14b')

def ollama_generate(prompt: str, system_prompt: str = SystemPrompt):
    """Generate response using local Ollama model"""
    payload = {
        "model": OLLAMA_MODEL,
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

# Create Gradio interface
def create_interface():
    """Create and return the Gradio chat interface"""
    return gr.ChatInterface(
        fn=ollama_generate,
        chatbot=gr.Chatbot(
            bubble_full_width=False,
            height=600
        ),
        additional_inputs=[
            gr.Textbox(
                label="System Prompt",
                value=SystemPrompt,
                lines=5
            )
        ]
    )

# Main function for running the app
def main():
    """Main function to run the AI chatbot application"""
    print("Starting AI Chatbot application...")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )

if __name__ == "__main__":
    main()