import streamlit as st
import json, os
import pickle
from utils import *

from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import asyncio
load_dotenv(".env.gpt4",override=True)

default_values = {
    "doc_processed": False,
    "uploaded_docs": None,
    "doc_chain": None,
    "list_of_messages": [],
    "temperature":0.5,
    "max_tokens":2000
    
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("Info Agent")

urls = ["https://elitebodyhome.com/",
"https://elitebodyhome.com/services/full-face-botox-dubai/",
"https://elitebodyhome.com/services/underarm-botox-dubai/",
"https://elitebodyhome.com/services/hyperhidrosis-botox/",
"https://elitebodyhome.com/services/botox-lip-flip/",
"https://elitebodyhome.com/services/masseter-botox/",
"https://elitebodyhome.com/services/allergan-botox/",
"https://elitebodyhome.com/services/dysport-botox/",
"https://elitebodyhome.com/services/nabota-botox/",
"https://elitebodyhome.com/services/nasolabial-fold-filler/",
"https://elitebodyhome.com/services/chin-fillers-dubai/",
"https://elitebodyhome.com/services/nose-fillers-dubai/",
"https://elitebodyhome.com/services/juvederm-fillers/",
"https://elitebodyhome.com/services/jawline-fillers-dubai/",
"https://elitebodyhome.com/services/laser-hair-removal/",
"https://elitebodyhome.com/services/laser-hair-removal-beard/",
"https://elitebodyhome.com/services/bikini-laser-hair-removal/",
"https://elitebodyhome.com/services/hair-generate-iv-drip/",
"https://elitebodyhome.com/services/vitamin-iv-drip-dubai/",
"https://elitebodyhome.com/skin-glowing-iv",
"https://elitebodyhome.com/services/cardio-support-iv-drip/"
]
if urls:
    with st.spinner("Processing Documents..."):
        #markdown_texts = asyncio.run(get_multiple_urls_text(urls))
        #combined_text = ""
        #home_page_content = markdown_texts[0]

        #for markdown_text in markdown_texts:
        #    if home_page_content in markdown_text:
        #        cleaned_text = markdown_text.replace(home_page_content, "")
        #        combined_text += cleaned_text + "\n"
        #    else:
        #        combined_text += markdown_text + "\n"
        #st.write(combined_text)
        
        #text_chunks = get_text_chunks(combined_text)
        #print(text_chunks)
        #vectorstore = get_vectorstore_faiss_huggingface(text_chunks)
        #with open('sample.pkl','wb') as f:
        #    pickle.dump(vectorstore, f)"""
        with open('sample.pkl','rb') as f:
            vectorstore=pickle.load(f)
        st.session_state.doc_chain = create_retrieval_chain_groq(vectorstore,st.session_state.temperature,st.session_state.max_tokens)
        st.session_state.doc_processed = True
else:
    st.session_state.doc_processed = False

                



 

clear_button = st.button("Clear Conversation", key="clear_history")

if clear_button:
    if st.session_state.doc_chain:
        st.session_state.doc_chain.memory.clear()
        st.session_state.list_of_messages = []


for i, message in enumerate(st.session_state.list_of_messages):
   
    if i%2 == 0:
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)
        
            

if prompt := st.chat_input("Prompt...."):
  
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):

        message_placeholder = st.empty()
        st_callback = StreamlitCallbackHandler(message_placeholder)
        response = st.session_state.doc_chain(
            {"question": prompt}, 
            callbacks=[st_callback]
            )

        
    message_placeholder.markdown(response["answer"])
    st.session_state.list_of_messages = response["chat_history"]
   

if os.environ.get("ENV") == "DEV" or os.environ.get("ENV") == "UAT":
    if st.session_state.list_of_messages != []:
        conversation_history = {"chat_history": []}
        for i, message in enumerate(st.session_state.list_of_messages):
            if i%2 == 0:
                conversation_history["chat_history"].append({"role": "user", "content": message.content})
            else:
                conversation_history["chat_history"].append({"role": "assistant", "content": message.content})

        st.download_button(
            label="Download History",
            data=json.dumps(conversation_history),
            file_name='conversation_history.json',
            mime='application/json',
            )