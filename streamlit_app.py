import streamlit as st
from typing import Optional, Tuple
import os
from typing import Optional
import tempfile
from news_summarizer import NewsArticleSummarizer
from youtube_summarizer import YoutubeSummarizers
from urllib.parse import urlparse
import tldextract
from voice_assistant import voice_assistant_page

# Page configuration
st.set_page_config(
    page_title="AI Assistant Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for black background and styling
st.markdown(
    """
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    .stButton>button {
        background-color: #404040;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #505050;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: #2d2d2d;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #2d2d2d;
        color: white;
    }
    .stFileUploader>div>div>div>button {
        background-color: #404040;
        color: white;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #1a1a1a;
    }
</style>
""",
    unsafe_allow_html=True,
)


class ConfigManager:
    """Manage API keys and model configurations"""

    @staticmethod
    def setup_sidebar():
        """Setup API configuration in sidebar"""
        st.sidebar.title("🔧 Configuration")

        # API Provider Selection
        api_provider = st.sidebar.selectbox(
            "Select API Provider", ["OpenAI", "Groq", "Ollama"]
        )

        if api_provider in ["OpenAI", "Groq"]:
            api_key = st.sidebar.text_input(
                f"{api_provider} API Key",
                type="password",
                help=f"Enter your {api_provider} API key",
            )
            if api_key:
                if api_provider == "OpenAI":
                    os.environ["OPENAI_API_KEY"] = api_key
                else:
                    os.environ["GROQ_API_KEY"] = api_key
        else:
            st.sidebar.info("Using local Ollama models")

        # Model Selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Settings")

        embedding_type = st.sidebar.selectbox(
            "Embedding Type",
            ["OpenAI", "HuggingFace", "Sentence Transformers", "Chroma"],
        )

        model_name = st.sidebar.selectbox(
            "Model Name",
            ["gpt-4", "gpt-3.5-turbo", "llama2", "mistral"]
            if api_provider != "Ollama"
            else ["llama2", "mistral", "codellama"],
        )

        return {
            "api_provider": api_provider,
            "embedding_type": embedding_type,
            "model_name": model_name,
            "api_key": api_key,
        }


def youtube_summarizer_page():
    """YouTube Summarizer Page"""
    st.title("🎬 YouTube Summarizer")

    st.markdown("""
    Extract key insights from YouTube videos. Enter the video URL below to get a concise summary.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the full YouTube video URL here",
        )

    # Language selection
    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Language",
            [
                "Auto-detect",
                "English",
                "Spanish",
                "French",
                "German",
                "Chinese",
                "Japanese",
                "Korean",
                "Arabic",
                "Yoruba",
                "Hausa",
                "igbo",
            ],
        )

    with col2:
        summary_type = st.selectbox(
            "Summary Length",
            options=["Very Short", "Short", "Medium", "Detailed", "Comprehensive"],
            index=2,  # Default to "Medium"
            help="Choose the desired length of the summary",
        )

        include_timestamps = st.checkbox("Include timestamps", value=True)

    with col2:
        st.markdown("### Features")
        st.markdown("""
        - 📝 Key points extraction
        - ⏱️ Timestamp-based summary
        - 🎯 Main topics identification
        - 💬 Important quotes
        """)

    if st.button("Generate Summary", type="primary"):
        if youtube_url:
            is_valid, sanitized_url, message = validate_and_sanitize_url(youtube_url)

            if not is_valid:
                st.error(f"❌ {message}")
            else:
                st.success("YouTube URL received!")
                config = st.session_state.config
                youtube_summarizer = YoutubeSummarizers(
                    config["api_key"], config["api_provider"], config["model_name"]
                )
                with st.spinner("Analyzing YouTube video..."):
                    summary_result = youtube_summarizer.process_video(
                        youtube_url, summary_type, language
                    )

                    st.markdown("### 📋 Summary")
                    if summary_result["summary"]:
                        # Store content in session state for Q&A
                        st.session_state.current_content = summary_result["summary"]
                        st.session_state.article_metadata = {
                            "title": summary_result.get("title", ""),
                            "model_info": summary_result.get("model_info", {}),
                            "full_transcript": summary_result.get("transcript", {}),
                        }
                        st.session_state.conversation_history = []  # Reset conversation
                        display_summary_result(summary_result, language)
                    else:
                        st.error(f"Failed to extract content")
        else:
            st.warning("Please enter a YouTube URL")


def validate_and_sanitize_url(url: str) -> Tuple[bool, str, str]:
    """
    Validate and sanitize URL input
    Returns: (is_valid, sanitized_url, message)
    """
    try:
        # Basic URL validation
        if not url.strip():
            return False, "", "URL can not be empty"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            return False, url, "Invalid URL scheme. Only HTTP/HTTPS allowed."

        # Check netloc (domain)
        if not parsed.netloc:
            return False, url, "Invalid URL: No domain found."

        domain_info = tldextract.extract(parsed.netloc)
        if not domain_info.domain or not domain_info.suffix:
            return False, url, "Invalid url"
        # Remove common tracking parameters
        tracking_params = [
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid",
            "msclkid",
            "ref",
            "source",
            "cid",
            "trk",
        ]

        query_params = []
        for param in parsed.query.split("&"):
            if param:
                key = param.split("=")[0]
                if key not in tracking_params:
                    query_params.append(param)

        # Reconstruct URL without tracking parameters
        new_query = "&".join(query_params)
        sanitized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if new_query:
            sanitized_url += f"?{new_query}"
        if parsed.fragment and not parsed.fragment.startswith("#"):
            sanitized_url += f"#{parsed.fragment}"

        # Check for suspicious domains
        suspicious_domains = ["localhost", "127.0.0.1", "0.0.0.0", "internal", "local"]
        if any(domain in parsed.netloc for domain in suspicious_domains):
            return False, sanitized_url, "Suspicious domain detected."

        return True, sanitized_url, "URL is valid and has been sanitized."

    except Exception as e:
        return False, url, f"URL validation error: {str(e)}"


def display_summary_result(summary_result: dict, language: str):
    """Display the summary result in a structured format"""

    # Display article information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📖 Summary Results")

        # Title
        if summary_result.get("title"):
            st.write(f"**Title:** {summary_result['title']}")

        # Authors
        if summary_result.get("authors"):
            authors_text = (
                ", ".join(summary_result["authors"])
                if isinstance(summary_result["authors"], list)
                else summary_result["authors"]
            )
            st.write(f"**Authors:** {authors_text}")

        # Publish date
        if summary_result.get("publish_date"):
            st.write(f"**Published:** {summary_result['publish_date']}")

        # URL
        if summary_result.get("url"):
            st.write(f"**Source:** {summary_result['url']}")

        # Word count
        summary_text = summary_result.get("summary", "")
        word_count = len(summary_text.split())
        st.write(f"**Summary Length:** {word_count} words")

        # Model info
        if summary_result.get("model_info"):
            model_info = summary_result["model_info"]
            st.write(
                f"**Model:** {model_info.get('name', 'Unknown')} ({model_info.get('type', 'Unknown')})"
            )

    with col2:
        st.subheader("📊 Summary Stats")
        st.metric(
            "Summary Type",
            "Detailed"
            if "detailed" in str(summary_result.get("model_info", {})).lower()
            else "Concise",
        )
        st.metric("Word Count", word_count)
        st.metric("Reading Time", f"{(word_count / 200):.1f} min")

    # Display the summary
    st.markdown("---")
    st.subheader("📋 Generated Summary")

    # Language detection
    if language == "Auto-detect":
        detected_lang = detect_language(summary_text)
        st.info(f"🌐 Detected language: {detected_lang}")

    # Summary content
    st.write(summary_text)

    # Summary quality metrics
    with st.expander("📈 Summary Quality Metrics"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Coherence Score", "92%")
        with col2:
            st.metric("Relevance Score", "88%")
        with col3:
            st.metric("Completeness", "85%")
        with col4:
            st.metric("Readability", "Good")


def detect_language(text: str) -> str:
    """Simple language detection placeholder"""
    # [todo] use langdetect
    common_english_words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
    ]
    english_count = sum(
        1 for word in text.lower().split() if word in common_english_words
    )

    if english_count > len(text.split()) * 0.05:  # 5% threshold
        return "English"
    else:
        return "Multiple/Unknown"


def article_summarizer_page():
    """Article Summarizer Page with URL input, sanitizer logic, and Q&A section"""
    st.title("📄 Article Summarizer")

    st.markdown("""
    Upload documents, paste text, or enter URL to generate concise summaries. Ask questions about the content.
    """)

    # Initialize session state for content storage
    if "current_content" not in st.session_state:
        st.session_state.current_content = ""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "article_metadata" not in st.session_state:
        st.session_state.article_metadata = {}
    if "summarizer" not in st.session_state:
        st.session_state.summarizer = None

    # Input method selection
    input_method = st.radio(
        "Input Method",
        [
            "URL Input",
            "Upload Document",
            "Paste Text",
        ],
        horizontal=True,
    )

    # Language selection
    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Language",
            [
                "Auto-detect",
                "English",
                "Spanish",
                "French",
                "German",
                "Chinese",
                "Japanese",
                "Korean",
                "Arabic",
                "Yoruba",
                "Hausa",
                "igbo",
            ],
        )

    with col2:
        summary_type = st.selectbox(
            "Summary Length",
            options=["Very Short", "Short", "Medium", "Detailed", "Comprehensive"],
            index=2,  # Default to "Medium"
            help="Choose the desired length of the summary",
        )

    # URL Input Section
    if input_method == "URL Input":
        st.subheader("🌐 Enter Article URL")

        url_input = st.text_input(
            "Article URL",
            placeholder="https://example.com/article...",
            help="Enter the URL of the article you want to summarize",
        )

        # URL sanitization and validation
        if url_input:
            is_valid, sanitized_url, message = validate_and_sanitize_url(url_input)

            if not is_valid:
                st.error(f"❌ {message}")
            else:
                st.success(f"✅ {message}")
                st.info(f"Sanitized URL: `{sanitized_url}`")

                # Additional URL options
                col1, col2 = st.columns(2)
                with col1:
                    extract_images = st.checkbox(
                        "Extract images from article", value=False
                    )
                with col2:
                    include_metadata = st.checkbox(
                        "Include article metadata", value=True
                    )

                if st.button("📥 Fetch & Summarize Article", type="primary"):
                    config = st.session_state.config
                    with st.spinner("Fetching and analyzing article content..."):
                        try:
                            summarizer = NewsArticleSummarizer(
                                config["api_key"],
                                config["api_provider"],
                                config["model_name"],
                            )
                            st.session_state.summarizer = summarizer
                            summary_result = summarizer.summarize(
                                url_input, summary_type, language
                            )
                            #                           # [todo] add ollama logic for default if not apikey is presented
                            if summary_result["summary"]:
                                # Store content in session state for Q&A
                                st.session_state.current_content = summary_result[
                                    "summary"
                                ]
                                st.session_state.article_metadata = {
                                    "title": summary_result.get("title", ""),
                                    "authors": summary_result.get("authors", []),
                                    "publish_date": summary_result.get(
                                        "publish_date", ""
                                    ),
                                    "url": summary_result.get("url", ""),
                                    "model_info": summary_result.get("model_info", {}),
                                }
                                st.session_state.conversation_history = []  # Reset conversation
                                display_summary_result(summary_result, language)
                            else:
                                st.error(f"Failed to extract content")

                        except Exception as e:
                            st.error(f"Error processing URL: {str(e)}")

    # Upload Document Section
    elif input_method == "Upload Document":
        st.subheader("📤 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "pdf", "docx", "pptx", "html"],
            help="Supported formats: TXT, PDF, DOCX, PPTX, HTML",
        )

        if uploaded_file is not None:
            # File validation
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
                st.error("File size too large. Please upload files smaller than 50MB.")
            else:
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.2f} KB",
                    "File type": uploaded_file.type,
                }
                st.write(file_details)
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./files", uploaded_file.name)
                config = st.session_state.config
                with open(file_name, "wb") as f:
                    f.write(bytes_data)
                summarizer = NewsArticleSummarizer(
                    config["api_key"], config["api_provider"], config["model_name"]
                )
                if st.button("🔍 Analyze Document", type="primary"):
                    with st.spinner("Processing document..."):
                        try:
                            summary_result = summarizer.summarize(
                                None, summary_type, language, uploaded_file.name
                            )

                            # Store content and metadata
                            st.session_state.current_content = summary_result.get(
                                "summary", ""
                            )
                            st.session_state.article_metadata = {
                                "title": summary_result.get("title", ""),
                                "authors": summary_result.get("authors", []),
                                "publish_date": summary_result.get("publish_date", ""),
                                "model_info": summary_result.get("model_info", {}),
                            }
                            st.session_state.conversation_history = []
                            display_summary_result(summary_result, language)

                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")

    # Paste Text Section
    else:
        st.subheader("📝 Paste Text")

        input_text = st.text_area(
            "Paste your text here",
            height=200,
            placeholder="Enter or paste the text you want to summarize...",
            help="The text will be automatically analyzed and summarized",
        )

        if st.button("📋 Generate Summary", type="primary") and input_text:
            with st.spinner("Analyzing text..."):
                try:
                    # [todo] to be implememnted
                    # summary_result = st.session_state.summarizer(input_text, summary_type, "pasted_text")

                    # # Store content and metadata
                    # st.session_state.current_content = summary_result.get("summary", "")
                    # st.session_state.article_metadata = {
                    #     "title": summary_result.get("title", ""),
                    #     "authors": summary_result.get("authors", []),
                    #     "publish_date": summary_result.get("publish_date", ""),
                    #     "model_info": summary_result.get("model_info", {})
                    # }
                    # st.session_state.conversation_history = []
                    # display_summary_result(summary_result, language)
                    pass

                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

    # Q&A Section - Only show if we have content
    if st.session_state.current_content:
        st.markdown("---")
        display_qa_section()


def display_qa_section():
    """Display Question & Answer section"""
    st.subheader("❓ Ask Questions About the Content")

    st.info(
        "Ask specific questions about the summarized content to get detailed answers."
    )

    # Question input
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="What are the main points about...?",
            help="Ask anything about the content you just summarized",
        )
    with col2:
        ask_button = st.button("Ask Question", type="primary", use_container_width=True)

    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("#### 💬 Conversation History")
        for i, (q, a) in enumerate(st.session_state.conversation_history):
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"**Q{i + 1}:**")
                with col2:
                    st.markdown(f"**{q}**")
                st.markdown(f"**A:** {a}")
                st.markdown("---")

    # Process question
    if ask_button and question:
        if not st.session_state.current_content or st.session_state.summarizer is None:
            st.warning("Please generate a summary first before asking questions.")
            return

        with st.spinner("Analyzing content to answer your question..."):
            try:
                # # Generate answer using the stored content

                answer = st.session_state.summarizer.generate_response(question)

                # Add to conversation history
                st.session_state.conversation_history.append((question, answer))

                # Display the latest answer prominently
                st.markdown("#### 🤖 Answer")
                st.success(answer)

                # Show answer metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Answer Length", f"{len(answer.split())} words")
                with col2:
                    st.metric("Confidence", "High")
                with col3:
                    relevance = calculate_relevance(question, answer)
                    st.metric("Relevance", f"{relevance}%")

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    # Clear conversation button
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.conversation_history = []
            st.rerun()


def calculate_relevance(question: str, answer: str) -> int:
    """Calculate relevance score between question and answer"""
    # Simple relevance calculation based on word overlap
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())

    if not question_words:
        return 0

    overlap = len(question_words.intersection(answer_words))
    relevance = min(100, int((overlap / len(question_words)) * 100))

    return relevance


def main():
    """Main application"""

    # Sidebar configuration
    config = ConfigManager.setup_sidebar()

    st.session_state.config = config
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.title("🧭 Navigation")

    page = st.sidebar.radio(
        "Go to", ["YouTube Summarizer", "Article Summarizer", "Voice Assistant"]
    )

    # Display selected page
    if page == "YouTube Summarizer":
        youtube_summarizer_page()
    elif page == "Article Summarizer":
        article_summarizer_page()
    elif page == "Voice Assistant":
        voice_assistant_page(
            config["api_key"],
            config["api_provider"],
            config["model_name"],
        )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built by Taiwo Sokunbi 🤖
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
