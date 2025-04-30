import streamlit as st
from langchain.prompts import PromptTemplate
from utils import create_llm_chain_groq,get_url_text
import asyncio
from dotenv import load_dotenv
load_dotenv(".env.groq", override=True)
st.title("General Inquiry Agent")

# Define the source text
url="https://elitebodyhome.com/"

source_text = asyncio.run(get_url_text(url))



# Create the LLMChain
if user_query := st.chat_input("Enter your Query here:"):
    st.markdown(user_query)
    Prompt = '''
    You are a general clinic enquiry assistant. Your role is to engage in a human-like conversation with the user and provide accurate and helpful responses to their queries about the clinic. Use the provided source file{source_text} to assist in answering the user query{user_query}. If you do not know the answer to a question, respond politely by saying, "I'm sorry, I don't know the answer to that. Please contact the clinic for more information."
    '''
    PROMPT_TEMPLATE = PromptTemplate(template=Prompt, input_variables=["source_text","user_query"])
    llm_chain=create_llm_chain_groq(PROMPT_TEMPLATE, temperature=0.5, max_tokens=4096)
    llm_out=llm_chain.run(source_text=source_text,user_query=user_query)

    st.write("Chatbot response:", llm_out)
