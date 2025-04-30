
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain,LLMChain
import asyncio
from langchain_groq import ChatGroq
import ssl
import os
from crawl4ai import *
from langchain.embeddings import OllamaEmbeddings
ssl._create_default_https_context = ssl._create_unverified_context
load_dotenv(".env.groq",override=True)

async def get_url_text(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return result.markdown


async def get_multiple_urls_text(urls):
    tasks = [get_url_text(url) for url in urls]
    return await asyncio.gather(*tasks)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=19000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore_faiss(text_chunks):
    embeddings = create_embedding_model_ollama()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def create_embedding_model_ollama():
    ssl._create_default_https_context = ssl._create_unverified_context
    embeddings = OllamaEmbeddings(model="llama3.1")
    return embeddings

def create_retrieval_chain_groq(vectorstore, temperature=0.5, max_tokens=3000):
    llm = create_llm_model_groq(temperature, max_tokens, streaming=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

def create_llm_model_groq(temperature, max_tokens, streaming=False):
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=temperature
)
    return llm
def create_llm_chain_groq(prompt_template, temperature, max_tokens):
    llm = create_llm_model_groq(temperature=temperature, max_tokens=max_tokens)
    return LLMChain(llm=llm, prompt=prompt_template)
