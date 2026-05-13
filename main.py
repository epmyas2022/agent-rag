import os

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import asyncio
from colorama import Fore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader



def read_pdf(file_path: str) -> str:

    if not os.path.exists(file_path):
        return None
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


async def main():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    document = read_pdf("document.pdf")

    if not document:
        print(f"{Fore.RED}Error: Could not read the PDF file.{Fore.RESET}")
        exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    chunks = splitter.split_text(document)

    model_embeddings = SentenceTransformer("BAAI/bge-m3")

    embeddings = model_embeddings.encode(chunks)

    question = sys.argv[1]

    question_embedding = model_embeddings.encode([question])

    vector_similary = cosine_similarity(question_embedding, embeddings)[0]

    index = np.argmax(vector_similary)

    print(f"\n({Fore.BLUE}Context: {chunks[index]}{Fore.RESET})")

    system_prompt = f"""
    # Rol: 
    Eres un asistente y debes responder usando el contexto

    ## Reglas

    - No debes agregar información que no esté en el contexto
    - Si no sabes la respuesta, responde "No lo sé"


    Responde con este contexto:
    {chunks[index]}
    """

    agent = create_agent(
        model="ollama:qwen3.5:4b",
        tools=[get_weather],
        system_prompt=system_prompt,
    )

    print("\n\n\nPregunta:", question)

    async for chunk, metadata in agent.astream(
        {"messages": [{"role": "user", "content": question}]}, stream_mode="messages"
    ):

        if isinstance(chunk, AIMessageChunk):
            print(Fore.YELLOW + chunk.content + Fore.RESET, end="", flush=True)


asyncio.run(main())
