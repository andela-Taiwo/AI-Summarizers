import os
import yt_dlp
from typing import Dict, List
import whisper
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import conversational_retrieval
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel:
    def __init__(
        self,
        api_key=os.getenv("OPENAI_API_KEY"),
        model_type: str = "openai",
    ):
        self.model_type = model_type

        if model_type.strip().lower() == "openai":
            self.embedding_fcn = OpenAIEmbeddings(
                model="text-embedding-3-small", api_key=api_key
            )
        elif model_type.strip().lower() == "chroma":
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embedding_fcn = HuggingFaceEmbeddings()
        elif model_type.strip().lower() == "nomic":
            from langchain.embeddings import OllamaEmbeddings

            self.embedding_fcn = OllamaEmbeddings(
                model="nomic-embed_text", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")


class LLMModel:
    def __init__(
        self, api_key: str, model_type: str = "openai", model_name: str = "gpt-4"
    ):
        self.model_type = model_type
        self.model_name = model_name
        if model_type.strip().lower() == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
            self.llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=api_key)
        elif model_type.strip().lower() == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")


class YoutubeSummarizers:
    def __init__(
        self,
        api_key: str,
        llm_type: str = "openai",
        llm_model_name: str = "gpt-4",
        embedding_type="openai",
    ):
        os.environ.update()
        self.embedding_model = EmbeddingModel(
            api_key,
            embedding_type,
        )
        self.llm_model = LLMModel(
            api_key,
            llm_type,
            llm_model_name,
        )
        self.whisper_model = whisper.load_model("base")

    def get_model_info(self) -> Dict:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_name": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str) -> tuple[str, str]:
        print("===== DOWNLOADING VIDEO ======")
        ydl_options = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "download/%(title)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
            video_title = info.get("title", "Unknown Title")
            return audio_path, video_title

    def transcribe_audio(self, audio_path: str) -> str:
        print("====== TRANSCRIBING AUDIO USING WHISPER =======")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, video_tittle: str) -> List[Document]:
        print("====== CREATING DOCUMENTS =======")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(text)
        return [
            Document(page_content=text_chunk, metadata={"source": video_tittle})
            for text_chunk in texts
        ]

    def creat_vector_store(self, documents: List[Document]) -> Chroma:
        print(
            f"Creating vector store using {self.embedding_model.model_type} embeddings..."
        )

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fcn,
            collection_name=f"youtube_summary_{self.embedding_model.model_type}",
        )

    def generate_summary(
        self,
        documents: List[Document],
        summary_type: str = "detailed",
        language: str = "english",
    ) -> str:
        map_prompt = ChatPromptTemplate.from_template(
            """Write a concuse summary of the following transcript section:
            "{text}"
            CONCISE SUMMARY:
            """
        )
        combine_prompt = ChatPromptTemplate.from_template(
            """Write a detailed summary of the following video transcript sections:
            "{text}"
            
            Include"
            - Main topics and key points
            -Important details and examples
            - Any conclusion
            
            DETAILED SUMMARY:
            """
        )

        # if summary_type == "detailed":
        map_prompt_template = f"""Write a detailed summary of the following text in {language}. 
        Focus on the main points and key insights:
        "{{text}}"
        {summary_type.upper()} SUMMARY IN {language.upper()}:"""
        combine_prompt_template = f"""Write a detailed summary in {language} that combines the previous summaries:
        "{{text}}"
        FINAL {summary_type.upper()} SUMMARY IN {language.upper()}:"""
        # else:
        #     map_prompt_template = f"""Write a concise summary of the following text in {language}:
        #     "{{text}}"
        #     CONCISE SUMMARY IN {language.upper()}:"""
        #     combine_prompt_template = f"""Write a concise summary in {language} that combines the previous summaries:
        #     "{{text}}"
        #     FINAL CONCISE SUMMARY IN {language.upper()}:"""

        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )
        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
        )
        return summary_chain.invoke(documents)

    def setup_qa_chain(self, vector_store: Chroma):
        """Set up question-answering chain"""

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
                # MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm_model.llm, vector_store.as_retriever(), contextualize_q_prompt
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
                # MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm_model.llm, qa_prompt
        )

        # 3. Combine into retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return rag_chain

    def process_video(
        self, url: str, summary_type: str = "concise", language: str = "english"
    ) -> Dict:
        try:
            os.makedirs("downloads", exist_ok=True)

            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents, summary_type, language)
            vector_store = self.creat_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)
            os.remove(audio_path)
            print(summary)
            return {
                "summary": summary["output_text"],
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }
        except Exception as e:
            print(f"Error processing videoL {str(e)}")
            return None


def main():
    urls = [
        "https://www.youtube.com/watch?v=dHLXfm9TKac",
        "https://www.youtube.com/watch?v=NGC2PQtfXe8",
    ]
    print("\n Available LLM Models:")
    print("1. OpenAI GPT-5")
    print("2. Ollama Llam3.2")

    llm_choice = input("Choose LLM model (1/2): ").strip()

    print("\nAvailable Embeddings:")
    print("1. OpenAI")
    print("2. Chroma Default")
    print("3. Nomic (via Ollama)")

    embedding_choice = input("Choose embeddings (1/2/3): ").strip()

    llm_type = "openai" if llm_choice == "1" else "ollama"
    llm_model_name = "gpt-5-mini" if llm_choice == "1" else "llama3.2"

    if embedding_choice == "1":
        embedding_type = "openai"
    elif embedding_choice == "2":
        embedding_type = "chroma"
    else:
        embedding_type = "nomic"

    try:
        summarizer = YoutubeSummarizers(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
            embedding_type=embedding_type,
        )
        print("igooorororo")
        model_info = summarizer.get_model_info()
        print("\n Current Configuration")
        print(f"LLM: {model_info['llm_type']} ({model_info['llm_name']})")
        print(f"Embeddings: {model_info['embedding_type']}")

        url = input("\nEnter YouTube URL: ")
        print(f"\nProcesing video...")
        result = summarizer.process_video(url)

        if result:
            print(f"\nVideo Title: {result['title']}")
            print("\nSummary:")
            print(result["summary"])

            print("\nYou now ask questions about video (type 'quit)' to exist")
            while True:
                query = input("\nYour question: ").strip()
                if query.lower() == "quit":
                    break

                if query:
                    response = result["qa_chain"].invoke({"input": query})
                    print(f"Result: {result['qa_chain']}")

                    print("\nAnswer:", response["answer"])

            if input("\nWant to see the full transcript? (y/n): ").strip() == "y":
                print("\nFull Transcript: ")
                print(result["full_transcript"])
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure required models and APIs are properly configured.")


if __name__ == "__main__":
    main()
