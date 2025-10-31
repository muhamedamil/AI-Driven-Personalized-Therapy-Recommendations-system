"""
Module: knowledge_base.py
Description:
    Provides an asynchronous service for loading, processing, chunking,
    and embedding knowledge base documents into a vector store using LangChain.

    This includes:
    - Reading `.txt` and `.pdf` files (with fallback OCR for image-based PDFs)
    - Token-based chunking for efficient embedding
    - Deduplication against existing vector store entries
    - Embedding and storing using the configured `VectorStoreService`
Created: 2025-06-10
Last Modified: 2025-07-08
"""

import logging
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from rag.vector_store import VectorStoreService

import fitz  # PyMuPDF
from tqdm.asyncio import tqdm
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import aiofiles

# Logger Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs/knowledge_base.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class KnowledgeBaseService:
    """
    A service for managing and processing documents into a RAG (Retrieval-Augmented Generation) knowledge base.

    Responsibilities:
    - Load `.txt` or `.pdf` files (including OCR fallback)
    - Split text content into token-based chunks
    - Store content into a vector database via the VectorStoreService

    Attributes:
        vector_store_service (VectorStoreService): Handles vector DB operations.
    """

    def __init__(self, collection_name: str):
        self.vector_store_service = VectorStoreService(collection_name)
        logger.info(f"Initialized KnowledgeBaseService for collection: {collection_name}")

    async def load_text_file(self, file_path: Path) -> List[Document]:
        """
        Load content from a plain text file into a LangChain-compatible Document.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            List[Document]: A list containing a single Document, or empty if the file is unreadable or empty.
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            if not content.strip():
                logger.warning(f"Text file is empty: {file_path}")
                return []
            logger.info(f"Loaded text file: {file_path}")
            return [Document(page_content=content, metadata={"source": str(file_path)})]
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {str(e)}")
            return []

    async def ocr_pdf_file(self, file_path: Path) -> str:
        """
        Perform OCR on a PDF file (fallback if standard text extraction fails).

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            str: Extracted text content.
        """
        try:
            images = convert_from_path(str(file_path))
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {str(e)}")
            return ""

    async def load_pdf_file(self, file_path: Path) -> List[Document]:
        """
        Attempt to extract text from a PDF file. Falls back to OCR if standard parsing fails.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            List[Document]: A list containing a single Document with extracted content, or empty list on failure.
        """
        try:
            doc = fitz.open(file_path)
            content = "".join([page.get_text() for page in doc])
            if not content.strip():
                logger.warning(f"Standard text extraction failed: {file_path}. Trying OCR...")
                content = await self.ocr_pdf_file(file_path)

            if not content.strip():
                logger.warning(f"OCR also found empty content in: {file_path}")
                return []

            logger.info(f"Loaded PDF file (text or OCR): {file_path}")
            return [Document(page_content=content, metadata={"source": str(file_path)})]
        except Exception as e:
            logger.error(f"Failed to load PDF file {file_path}: {str(e)}")
            return []

    async def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """
        Load all supported documents from a folder (.txt and .pdf).

        Args:
            folder_path (str): Folder containing the documents.

        Returns:
            List[Document]: All loaded documents.
        """
        folder = Path(folder_path)
        documents = []

        if not folder.exists() or not folder.is_dir():
            logger.error(f"The folder {folder} does not exist or is not a directory.")
            return documents

        for file_path in folder.glob("*"):
            if file_path.suffix.lower() == ".txt":
                docs = await self.load_text_file(file_path)
                documents.extend(docs)
            elif file_path.suffix.lower() == ".pdf":
                docs = await self.load_pdf_file(file_path)
                documents.extend(docs)
            else:
                logger.warning(f"Skipping unsupported file type: {file_path}")

        logger.info(f"Loaded {len(documents)} documents from folder: {folder}")
        return documents

    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks using token-based logic.

        Args:
            documents (List[Document]): Raw documents.

        Returns:
            List[Document]: Chunked documents with inherited metadata.
        """
        text_splitter = TokenTextSplitter(chunk_size=384, chunk_overlap=20)
        chunked_docs = []

        for doc in documents:
            try:
                chunks = text_splitter.split_text(doc.page_content)
                chunked_docs.extend([
                    Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks
                ])
                logger.info(f"Chunked document into {len(chunks)} chunks from source: {doc.metadata['source']}")
            except Exception as e:
                logger.error(f"Failed to chunk document: {str(e)}")

        return chunked_docs

    async def embed_and_store_documents(self, documents: List[Document]) -> None:
        """
        Embed and store documents in the vector database.

        Args:
            documents (List[Document]): Chunked documents to store.
        """
        if not documents:
            logger.warning("No documents to embed and store.")
            return
        try:
            await self.vector_store_service.add_documents(documents)
            logger.info(f"Successfully stored {len(documents)} chunks in the vector store.")
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise

    async def process_documents(self, folder_path: str) -> None:
        """
        Full document processing pipeline:
        - Loads documents from folder
        - Filters out already-ingested sources
        - Chunks text
        - Embeds and stores into vector store

        Args:
            folder_path (str): Folder containing the knowledge base files.
        """
        if not Path(folder_path).exists():
            logger.error(f"Provided folder path does not exist: {folder_path}")
            return

        documents = await self.load_documents_from_folder(folder_path)
        if not documents:
            logger.warning("No valid documents found in the folder.")
            return

        deduped_documents = []
        for doc in documents:
            exists = await self.vector_store_service.has_document(doc.metadata["source"])
            if exists:
                logger.info(f"Skipping existing document: {doc.metadata['source']}")
            else:
                deduped_documents.append(doc)

        if not deduped_documents:
            logger.info("All documents already exist in the vector store. Nothing new to process.")
            return

        chunked_docs = await self.chunk_documents(deduped_documents)
        await self.embed_and_store_documents(chunked_docs)
        logger.info("Document processing completed successfully.")


__all__ = ["KnowledgeBaseService"]
