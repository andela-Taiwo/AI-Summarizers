# AI Summarizers 🚀
A powerful Streamlit application that provides multiple AI-powered summarization tools including YouTube video summarization, article/document summarization, and voice-assisted RAG (Retrieval-Augmented Generation) capabilities.

### Features ✨
#### 🎬 YouTube Summarizer
    Extract key insights from YouTube videos

    Generate concise summaries with timestamps

    Support for various video lengths and content types

#### 📄 Article & Document Summarizer
    Multiple Input Methods:

    URL input for web articles

    Document upload (PDF, TXT, MD, DOCX)

    Direct text input

    Smart Language Detection with multi-language support

    Adjustable Summary Length from concise to comprehensive

    Q&A System - Ask questions about summarized content

#### 🎤 Voice Assistant RAG
    Voice Recording - Record and transcribe audio queries

    Audio File Upload - Process existing audio files

    Text-to-Speech - Convert responses to natural speech

    Conversational Memory - Maintains context across interactions

🔧 Advanced Features
    Multiple AI Providers: OpenAI, Groq, and Ollama support

    Vector Database: ChromaDB for efficient document retrieval

    Customizable Models: Choose between different LLMs and embedding types

    Session Management: Persistent chat history and document processing

### Installation 🛠️
#### Prerequisites
Python 3.13+

UV package manager

#### Quick Start
Clone the repository:

```bash
git clone https://github.com/andela-Taiwo/AI-Summarizers.git
cd AI-Summarizers
```
Install dependencies using UV:

```bash
uv sync
```

Set up environment variables:

```bash
cp .env.example .env
```
Edit .env with your API keys:

#### env
    OPENAI_API_KEY=your_openai_api_key_here
    ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here
    GROQ_API_KEY=your_groq_api_key_here
    

#### Run the application:

```bash
uv run streamlit run streamlit_app.py
```

#### Configuration ⚙️
    API Keys Required
        OpenAI API Key: For GPT models and embeddings

        ElevenLabs API Key: For text-to-speech functionality

        Groq API Key (Optional): For faster inference with Groq models

    Model Options
        OpenAI: GPT-4, GPT-3.5-Turbo, GPT-4o-mini

        Groq: Llama2, Mixtral, Gemma

        Ollama: Local models (Llama2, Mistral, etc.)

    Embedding Options
        OpenAI Embeddings
        HuggingFace 
        Sentence Transformers


#### Usage Guide 📖
    1. YouTube Summarization
        Go to "YouTube Summarizer" page

        Paste YouTube URL

        Select summary length and options

        Generate comprehensive video summary

    2. Article Summarization
        Access "Article Summarizer" page

        Choose input method (URL, Upload, or Paste Text)

        Select language and summary type

        Generate summary and ask follow-up questions

    3. Voice Assistant
        Visit "Voice Assistant" page

        Set up knowledge base first (required)

        Choose between recording or uploading audio

        Interact naturally with voice or text

Dependencies 📦
 - pyproject.toml


### Development 🧑‍💻
Local Development
```bash
# Install development dependencies
uv sync --dev
```
# Run with hot reload
```bash uv run streamlit run streamlit_app.py --server.runOnSave true
```

Environment Variables
    The application uses the following environment variables:

    OPENAI_API_KEY: For OpenAI models and embeddings

    ELEVEN_LABS_API_KEY: For voice generation

    GROQ_API_KEY: For Groq model inference (optional)

    OLLAMA_BASE_URL: For local Ollama models (optional)


#### Troubleshooting 🔧
    Common Issues
    "Database error: attempt to write a readonly database"

    Solution: The app automatically falls back to in-memory storage

    Check directory permissions if using persistent storage

    Audio recording not working

    Ensure microphone permissions are granted

    Check browser compatibility for audio recording

#### API Key errors

    Verify all required API keys are set in environment variables

    Check for typos in API keys

    Document processing failures

    Ensure documents are in supported formats

    Check file sizes (recommended: <50MB per file)

Performance Tips
Use Groq for faster inference with open-source models

Process documents in batches for large collections

Use "Clear Session" to free memory during extended use

Contributing 🤝
We welcome contributions! Please see our Contributing Guidelines for details.

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments 🙏
OpenAI for GPT models and embeddings

ElevenLabs for high-quality text-to-speech

Groq for ultra-fast inference

LangChain for the AI framework

Streamlit for the amazing web framework

Support 💬
If you encounter any issues or have questions:

Check the Troubleshooting section

Search existing GitHub Issues

Create a new issue with detailed information

Star History ⭐
https://api.star-history.com/svg?repos=andela-Taiwo/AI-Summarizers&type=Date

Made with ❤️ by Taiwo

If you find this project useful, please give it a ⭐ on GitHub!
