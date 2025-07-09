import os
from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Get the list of files in the gradio_client directory
try:
    import pkg_resources
    gradio_client_path = pkg_resources.resource_filename('gradio_client', '')
    data_files = []
    
    # Add types.json if it exists
    import os.path
    types_json_path = os.path.join(gradio_client_path, 'types.json')
    if os.path.exists(types_json_path):
        data_files.append(("gradio_client", [types_json_path]))
        
except ImportError:
    data_files = []

setup(
    name="ai-chatbot-app",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="AI Chatbot Application with Ollama Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "gradio>=3.0.0",
        "requests>=2.0.0",
        "modelscope-studio>=0.1.0",
        "ollama>=0.1.0"
    ],
    package_data={
        "": ["*.css", "*.toml", "*.json"]
    },
    entry_points={
        "console_scripts": [
            "ai-chatbot-app=app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
)