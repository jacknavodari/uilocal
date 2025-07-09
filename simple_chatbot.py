"""
AI Chatbot Application with Ollama Integration (Simple Version)
"""
import os
import json
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer

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
                            yield json.dumps({
                                "content": content,
                                "is_error": False
                            }) + '\n'
                    except json.JSONDecodeError:
                        continue
        
    except requests.exceptions.RequestException as e:
        yield json.dumps({
            "content": f"Error connecting to Ollama: {str(e)}",
            "is_error": True
        }) + '\n'

# Simple HTTP server to serve the chat interface
class ChatRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/chat':
            # Get prompt from request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            # Get response from Ollama
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Stream response
            for chunk in ollama_generate(post_data):
                self.wfile.write(chunk.encode())
                self.wfile.flush()
    
    def do_GET(self):
        if self.path == '/':
            # Serve simple HTML interface
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(SIMPLE_HTML.encode())

SIMPLE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #chat { height: 400px; border: 1px solid #ccc; overflow-y: scroll; padding: 10px; margin-bottom: 10px; background: #f9f9f9; }
        .message { margin: 10px 0; }
        .user { color: blue; }
        .assistant { color: green; }
        input[type="text"] { width: 80%; padding: 10px; }
        button { padding: 10px; }
    </style>
</head>
<body>
    <h1>AI Chatbot</h1>
    <div id="chat"></div>
    <form id="chat-form">
        <input type="text" id="prompt" placeholder="Enter your message...">
        <button type="submit">Send</button>
    </form>

    <script>
        const form = document.getElementById('chat-form');
        const promptInput = document.getElementById('prompt');
        const chat = document.getElementById('chat');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const prompt = promptInput.value;
            if (!prompt) return;

            // Add user message to chat
            chat.innerHTML += `<div class="message user">User: ${prompt}</div>`;
            
            // Clear input
            promptInput.value = '';
            
            // Send request to server
            const response = await fetch('/chat', {
                method: 'POST',
                body: prompt
            });
            
            // Stream response
            const reader = response.body.getReader();
            let result = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = new TextDecoder().decode(value);
                try {
                    const json = JSON.parse(chunk);
                    if (json.is_error) {
                        result = `<div class="message error">Error: ${json.content}</div>`;
                    } else {
                        result = `<div class="message assistant">Assistant: ${json.content}</div>`;
                    }
                    chat.innerHTML += result;
                    chat.scrollTop = chat.scrollHeight;
                } catch (e) {
                    console.error('Error parsing JSON:', e);
                }
            }
        });
    </script>
</body>
</html>
'''  # End of HTML

# Main function for running the app
def main():
    """Main function to run the AI chatbot application"""
    print("Starting AI Chatbot application...")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print("Server running at http://localhost:8080")
    
    # Create and start the server
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, ChatRequestHandler)
    httpd.serve_forever()

if __name__ == "__main__":
    main()