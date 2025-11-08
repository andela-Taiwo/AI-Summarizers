import os
from typing import Optional
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_classic.schema import Document
from newspaper import Article
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()


class NewsArticleSummarizer:
    def __init__(
        self,
        api_key: str = None,
        model_type: str = "openai",
        model_name: str = "gpt-4o-mini",
    ):
        """
        Initialize the summarizer with choice of model
        Args:
            api_key: OpenAI API key (required for OpenAI models)
            model_type: 'openai' or 'ollama'
            model_name: specific model name
        """
        self.model_type = model_type
        self.model_name = model_name

        # Setup LLM based on model type
        if model_type.lower().strip() == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI models")
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        elif model_type.lower().strip() == "ollama":
            # Using ChatOllama with proper configuration
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",  # Optional: for structured output
                timeout=120,  # Increased timeout for longer generations
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize text splitter for long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

    def create_documents(self, text: str) -> list[Document]:
        texts = self.text_splitter.split_text(text)
        docs = [Document(page_content=text) for text in texts]
        return docs

    def fetch_article(self, url: str) -> Optional[Article]:
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            print(f"Error fetching article: {str(e)}")
            return None

    def load_document(self, file):
        try:
            file_path = os.path.join("files", file)
            name, extension = os.path.splitext(file_path)

            if extension == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(file_path)
            elif extension == ".docx":
                from langchain_community.document_loaders import Docx2txtLoader

                loader = Docx2txtLoader(file_path)
            elif extension == ".txt":
                from langchain_community.document_loaders import TextLoader

                loader = TextLoader(file_path)
            else:
                print("Document format not supported!")
                return None

            print(f"Loading document: {file_path}")
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            return None

    def process_uploaded_file(self, uploaded_file_name):
        return self.load_document(uploaded_file_name)

    def summarize(
        self,
        url: str = None,
        summary_type: str = "detailed",
        language: str = "english",
        document_name: str = None,
    ) -> dict:
        try:
            docs = []
            article_data = {}
            content_source = ""

            # Get content from URL or document
            if url:
                article = self.fetch_article(url)
                if article:
                    docs = self.create_documents(article.text)
                    article_data = {
                        "title": getattr(article, "title", ""),
                        "authors": getattr(article, "authors", []),
                        "publish_date": getattr(article, "publish_date", ""),
                        "text": article.text,
                    }
                    content_source = "url"

            elif document_name:
                documents = self.load_document(document_name)
                if documents and len(documents) > 0:
                    # Combine all document pages
                    full_text = "\n".join([doc.page_content for doc in documents])
                    docs = self.create_documents(full_text)
                    article_data = {
                        "title": document_name,
                        "authors": [],
                        "publish_date": "",
                        "text": full_text,
                    }
                    content_source = "document"

            # Check if we successfully got content
            if not docs:
                return {"Error": "Failed to fetch article or load document"}

            # Create prompts based on summary type and language
            if summary_type == "detailed":
                map_prompt_template = f"""Write a detailed summary of the following text in {language}. 
                Focus on the main points and key insights:
                "{{text}}"
                DETAILED SUMMARY IN {language.upper()}:"""
                combine_prompt_template = f"""Write a detailed summary in {language} that combines the previous summaries:
                "{{text}}"
                FINAL DETAILED SUMMARY IN {language.upper()}:"""
            else:
                map_prompt_template = f"""Write a concise summary of the following text in {language}:
                "{{text}}"
                CONCISE SUMMARY IN {language.upper()}:"""
                combine_prompt_template = f"""Write a concise summary in {language} that combines the previous summaries:
                "{{text}}"
                FINAL CONCISE SUMMARY IN {language.upper()}:"""

            map_prompt = PromptTemplate(
                template=map_prompt_template, input_variables=["text"]
            )
            combine_prompt = PromptTemplate(
                template=combine_prompt_template, input_variables=["text"]
            )

            # Create and run the summarization chain
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=False,
            )

            summary_result = chain.invoke(docs)

            # Extract the summary text safely
            summary_text = ""
            if isinstance(summary_result, dict) and "output_text" in summary_result:
                summary_text = summary_result["output_text"]
            elif hasattr(summary_result, "output_text"):
                summary_text = summary_result.output_text
            else:
                summary_text = str(summary_result)

            return {
                "title": article_data.get("title", ""),
                "authors": article_data.get("authors", []),
                "publish_date": article_data.get("publish_date", ""),
                "summary": summary_text,
                "url": url,
                "source": content_source,
                "model_info": {"type": self.model_type, "name": self.model_name},
            }

        except Exception as e:
            print(f"Error summarizing: {str(e)}")
            return {"Error": f"Summarization failed: {str(e)}"}


def main():
    # Example of using both models
    url = "https://www.artificialintelligence-news.com/news/us-china-ai-chip-race-cambricons-first-profit-lands/"

    openai_summarizer = NewsArticleSummarizer(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_type="openai",
        model_name="gpt-4o-mini",  # Fixed: changed from "gpt-5-mini" to "gpt-4o-mini"
    )

    print("\nGenerating OpenAI Summary...")
    openai_summary = openai_summarizer.summarize(url, summary_type="detailed")

    # Check if summary was successful
    if "Error" in openai_summary:
        print(f"Error: {openai_summary['Error']}")
    else:
        print(f"Title: {openai_summary['title']}")
        print(
            f"Authors: {', '.join(openai_summary['authors']) if openai_summary['authors'] else 'Unknown'}"
        )
        print(f"Published: {openai_summary['publish_date']}")
        print(f"Summary:\n{openai_summary['summary']}")


if __name__ == "__main__":
    main()
