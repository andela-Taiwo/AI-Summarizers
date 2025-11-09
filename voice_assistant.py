import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI

from langchain_classic.chains import conversational_retrieval

from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import tempfile
from dotenv import load_dotenv
import os
from typing import List
from langchain_core.documents import Document

load_dotenv()


class DocumentProcessor:
    def __init__(self, provider):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        print(provider, "#" * 14)
        if provider.strip().lower() in ["openai"]:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        elif provider.strip().lower() in ["chroma", "groq"]:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings()
        elif provider.strip().lower() == "nomic":
            from langchain.embeddings import OllamaEmbeddings

            self.embeddings = OllamaEmbeddings(
                model="nomic-embed_text", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {provider}")

    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from different file types"""
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(
                directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
            ),
        }

        documents = []
        for file_type, loader in loaders.items():
            try:
                documents.extend(loader.load())
                print(f"Loaded {file_type} documents")
            except Exception as e:
                print(f"Error loading {file_type} documents: {str(e)}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(
        self, documents: List[Document], persist_directory: str = None
    ) -> Chroma:
        # Use provided directory or create temporary one
        if persist_directory is None:
            self.persist_directory = tempfile.mkdtemp(prefix="chroma_db_")
        else:
            self.persist_directory = persist_directory
            os.makedirs(self.persist_directory, exist_ok=True)

        try:
            print(f"Creating vector store in: {self.persist_directory}")

            # Create new vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

            # Try to persist
            try:
                vector_store.persist()
                print("Vector store persisted successfully")
            except Exception as persist_error:
                print(f"Warning: Could not persist vector store: {persist_error}")
                # Continue with in-memory version

            return vector_store

        except Exception as e:
            print(f"Error creating persistent vector store: {e}")
            # Fallback to in-memory
            print("Using in-memory vector store as fallback")
            return Chroma.from_documents(documents=documents, embedding=self.embeddings)


class VoiceGenerator:
    def __init__(self, api_key):
        self.client = ElevenLabs(api_key=api_key)
        # Default available voices
        self.available_voices = [
            "Rachel",
            "Domi",
            "Bella",
            "Antoni",
            "Elli",
            "Josh",
            "Arnold",
            "Adam",
            "Sam",
        ]
        self.default_voice = "Rachel"
        self.voice_ids = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Adam": "pNInz6obpgDQGcFmaJgB",
            "Sam": "yoZ06aMxZJJ28mfd3POQ",
        }

    def get_voice_id(self, voice_name: str = None) -> str:
        """Get voice ID from voice name"""
        selected_voice = voice_name or self.default_voice
        return self.voice_ids.get(selected_voice, self.voice_ids[self.default_voice])

    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        """Generate voice response"""
        try:
            voice_id = self.get_voice_id(voice_name)

            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name

        except Exception as e:
            print(f"Error generating voice response: {e}")
            return None

    def get_available_voices(self):
        """Get list of available voices from ElevenLabs API"""
        try:
            # Fetch actual voices from API
            voices_response = self.client.voices.get_all()
            voices = []
            for voice in voices_response.voices:
                voices.append(voice.name)
                # Update voice_ids mapping with actual IDs
                self.voice_ids[voice.name] = voice.voice_id
            return voices
        except Exception as e:
            print(f"Error fetching voices from API: {e}")
            # Fallback to default voices
            return self.available_voices


class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key, provider="openai", model_name="gpt-4o-mini"):
        self.whisper_model = whisper.load_model("base")
        self.vector_store = None
        self.qa_chain = None
        self.provider = provider
        self.model_name = model_name
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)
        self.chat_history = []  # Add chat history storage
        self._set_model()

    def _set_model(self):
        if self.provider.lower() == "openai":
            self.llm = ChatOpenAI(model_name=self.model_name, temperature=0.3)
            self.embeddings = OpenAIEmbeddings()
        elif self.provider.lower() == "groq":
            from groq import Groq

            self.llm = Groq(model_name=self.model_name, temperature=0.3)
            self.embeddings = OpenAIEmbeddings()
        elif self.provider.lower() == "ollama":
            from langchain.embeddings import OllamaEmbeddings

            self.embeddings = OllamaEmbeddings(
                model="nomic-embed_text", base_url="http://localhost:11434"
            )
            self.llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            # self.model_name = "llama3.2"

    def setup_vector_store(self, vector_store):
        """Initialize the vector store and QA chain"""
        self.vector_store = vector_store

        # 1. Create history-aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Given a chat history and the latest user question, 
            which might reference context in the chat history, formulate a standalone question 
            which can be understood without the chat history. Do NOT answer the question, 
            just reformulate it if needed and otherwise return it as is.""",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),  # Fixed: changed {query} to {input}
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm,  # Fixed: using self.llm directly
            self.vector_store.as_retriever(),
            contextualize_q_prompt,
        )

        # 2. Create QA chain with chat history
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.\n\n
            Context: {context}""",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt,
        )

        # 3. Combine into retrieval chain
        self.qa_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        recording = sd.rec(
            int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1
        )
        sd.wait()
        return recording

    def transcribe_audio_file(self, audio_file_path):
        """Transcribe audio file using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing audio file: {str(e)}")
            return None

    def transcribe_audio(self, audio_array):
        """Transcribe audio array using Whisper"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, audio_array, self.sample_rate)
            result = self.whisper_model.transcribe(temp_audio.name)
            os.unlink(temp_audio.name)
        return result["text"]

    def generate_response(self, query):
        """Generate response using RAG system"""
        if self.qa_chain is None:
            return "Error: Vector store not initialized"

        try:
            # Fixed: Use correct input format with "input" key and include chat_history
            response = self.qa_chain.invoke(
                {
                    "input": query,  # Fixed: changed "question" to "input"
                    "chat_history": self.chat_history,
                }
            )

            # Update chat history
            from langchain_core.messages import HumanMessage, AIMessage

            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response["answer"]))

            return response["answer"]
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """Convert text to speech"""
        return self.voice_generator.generate_voice_response(text, voice_name)

    def clear_chat_history(self):
        """Clear the conversation history"""
        self.chat_history = []


def setup_knowledge_base(provider="openai"):
    st.title("Knowledge Base Setup")

    doc_processor = DocumentProcessor(provider)

    uploaded_files = st.file_uploader(
        "Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "md"]
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()

            # Save uploaded files
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            try:
                # Process documents
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                # Create vector store
                vector_store = doc_processor.create_vector_store(
                    processed_docs, "knowledge_base"
                )

                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.knowledge_base_ready = True

                st.success(f"Processed {len(processed_docs)} document chunks!")

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
            finally:
                # Cleanup
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)


def process_audio_upload(assistant, selected_voice):
    """Handle audio file upload and processing"""
    st.subheader("🎵 Upload Audio File")

    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="Supported formats: WAV, MP3, M4A, OGG, FLAC",
    )

    if uploaded_audio is not None:
        # Display audio info
        st.audio(uploaded_audio, format=f"audio/{uploaded_audio.type.split('/')[-1]}")
        st.write(f"**File:** {uploaded_audio.name}")
        st.write(f"**Size:** {uploaded_audio.size / 1024:.2f} KB")
        st.write(f"**Type:** {uploaded_audio.type}")

        if st.button("Transcribe & Process Audio File"):
            with st.spinner("Processing uploaded audio..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}"
                    ) as temp_file:
                        temp_file.write(uploaded_audio.getvalue())
                        temp_file_path = temp_file.name

                    # Transcribe audio file
                    query = assistant.transcribe_audio_file(temp_file_path)

                    # Clean up temporary file
                    os.unlink(temp_file_path)

                    if query:
                        st.success("Audio transcribed successfully!")
                        st.write("**Transcribed Text:**", query)

                        # Generate response
                        with st.spinner("Generating response..."):
                            response = assistant.generate_response(query)
                            st.write("**Assistant Response:**", response)

                            # Convert to speech
                            with st.spinner("Converting to speech..."):
                                audio_file = (
                                    assistant.voice_generator.generate_voice_response(
                                        response, selected_voice
                                    )
                                )
                                if audio_file:
                                    st.audio(audio_file)
                                    # Clean up the temporary file after displaying
                                    try:
                                        os.unlink(audio_file)
                                    except:
                                        pass  # File might already be deleted
                                else:
                                    st.error("Failed to generate voice response")
                    else:
                        st.error("Failed to transcribe audio file")

                except Exception as e:
                    st.error(f"Error processing audio file: {str(e)}")


def process_voice_recording(assistant, selected_voice, duration):
    """Handle voice recording and processing"""
    st.subheader("🎤 Voice Recording")

    # Initialize session state for recording
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "last_transcription" not in st.session_state:
        st.session_state.last_transcription = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Recording", key="record_btn"):
            with st.spinner(f"Recording for {duration} seconds..."):
                audio_data = assistant.record_audio(duration)
                st.session_state.audio_data = audio_data
                st.session_state.last_transcription = ""
                st.session_state.last_response = ""
                st.success("Recording completed!")

    with col2:
        if st.button("Process Recording", key="process_btn"):
            if st.session_state.audio_data is None:
                st.error("Please record audio first!")
                return

            # Process recording
            with st.spinner("Transcribing..."):
                query = assistant.transcribe_audio(st.session_state.audio_data)
                st.session_state.last_transcription = query
                st.success("Transcription completed!")
                st.write("**You said:**", query)

            with st.spinner("Generating response..."):
                try:
                    response = assistant.generate_response(query)
                    st.session_state.last_response = response
                    st.success("Response generated!")
                    st.write("**Assistant:**", response)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    return

            with st.spinner("Converting to speech..."):
                audio_file = assistant.voice_generator.generate_voice_response(
                    response, selected_voice
                )
                if audio_file:
                    st.audio(audio_file)
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                else:
                    st.error("Failed to generate voice response")

    # Display current session data
    if st.session_state.last_transcription or st.session_state.last_response:
        st.markdown("---")
        st.subheader("Current Session")

        if st.session_state.last_transcription:
            st.write(f"**Last Transcription:** {st.session_state.last_transcription}")

        if st.session_state.last_response:
            st.write(f"**Last Response:** {st.session_state.last_response}")


def display_chat_history(assistant):
    """Display chat history"""
    if assistant.chat_history:
        st.subheader("💬 Conversation History")
        for i, message in enumerate(assistant.chat_history):
            if hasattr(message, "content"):
                role = "You" if message.type == "human" else "Assistant"
                st.markdown(f"**{role}:** {message.content}")
                if i < len(assistant.chat_history) - 1:
                    st.markdown("---")


def voice_assistant_page(api_key: str, provider, model_name):
    st.set_page_config(page_title="Voice RAG Assistant", layout="wide")
    if provider.lower() != "ollama" and not api_key:
        st.error(
            "Please set GROQ_API_KEY or OPENAI_API_KEY in your environment variables"
        )

    # Check for API keys
    elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
    openai_api_key = api_key

    # Initialize session state
    if "knowledge_base_ready" not in st.session_state:
        st.session_state.knowledge_base_ready = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "assistant" not in st.session_state:
        st.session_state.assistant = None

    if not elevenlabs_api_key:
        st.error("Please set ELEVEN_LABS_API_KEY in your environment variables")
        return

    # Navigation
    st.sidebar.subheader("⚙️ Voice")
    page = st.sidebar.radio("Go to", ["Setup Knowledge Base", "Voice Assistant"])

    if page == "Setup Knowledge Base":
        setup_knowledge_base()

    else:  # Voice Assistant page
        if (
            not st.session_state.knowledge_base_ready
            or st.session_state.vector_store is None
        ):
            st.error("Please setup knowledge base first!")
            st.info("Go to 'Setup Knowledge Base' to upload and process your documents")
            return

        st.title("🎤 Voice Assistant RAG System")

        # Initialize assistant only once
        if st.session_state.assistant is None:
            st.session_state.assistant = VoiceAssistantRAG(
                elevenlabs_api_key, provider, model_name
            )
            st.session_state.assistant.setup_vector_store(st.session_state.vector_store)

        assistant = st.session_state.assistant

        # Sidebar configuration
        st.sidebar.subheader("Voice Settings")

        # Voice selection
        try:
            available_voices = assistant.voice_generator.available_voices
            if available_voices:
                selected_voice = st.sidebar.selectbox(
                    "Select Voice",
                    available_voices,
                    index=(
                        available_voices.index("Rachel")
                        if "Rachel" in available_voices
                        else 0
                    ),
                )
            else:
                st.sidebar.warning("No voices available. Using default voice.")
                selected_voice = "Rachel"
        except Exception as e:
            st.sidebar.error(f"Error loading voices: {e}")
            selected_voice = "Rachel"

        # Recording duration
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

        # Clear chat history button
        if st.sidebar.button("Clear Chat History"):
            assistant.clear_chat_history()
            st.session_state.audio_data = None
            st.session_state.last_transcription = ""
            st.session_state.last_response = ""
            st.rerun()

        # Main content area with tabs
        tab1, tab2 = st.tabs(["🎤 Voice Recording", "🎵 Upload Audio File"])

        with tab1:
            process_voice_recording(assistant, selected_voice, duration)

        with tab2:
            process_audio_upload(assistant, selected_voice)

        # Display chat history
        display_chat_history(assistant)


if __name__ == "__main__":
    voice_assistant_page()
