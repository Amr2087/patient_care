# pip install streamlit langchain-core langgraph>0.2.27 langchain-groq
import streamlit as st
from pinecone import Pinecone
import getpass
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

st.title('Chat Bot App')

pinecone_api_key = "pcsk_5LPCzg_6A8fbo65hKuepW2rLi5kL4MyqftWzdVYGNVVfRKE6fCXyhUeAM6jSWW48886VWZ"

pc = Pinecone(api_key=pinecone_api_key)
index_name = "langchain-test-index"
index = pc.Index(index_name)



embeddings_model = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/Arabic-labse-Matryoshka")
os.environ["GROQ_API_KEY"] = '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'


llm = ChatGroq(model="llama-3.1-70b-versatile")
vector_store = PineconeVectorStore(index=index, embedding=embeddings_model)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)

system_prompt = (
    "You are a Q&A chatbot that acts as a Nurse to help patients with their problems and you answer the user query from the context you have only, if the context dosent have the answer say عفوا لا استطيع المساعدة, make your answer as simplfied as possible so any patient can understand and follow your advice easily. answer only in arabic, dont ever answer a question that the answer to is not provided in the context if will help patient just say Cant help You."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

placeholder = "How can i help you?"

if "msgs" not in st.session_state:
    st.session_state.msgs = []


for message in st.session_state.msgs:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if query := st.chat_input(placeholder):
    st.session_state.msgs.append({'role': 'user', 'content': query})

    with st.chat_message(query):
        st.markdown(query)

    # Render the system prompt
    system_prompt_rendered = system_prompt.format(context="")  # Replace `""` with actual context if necessary

    full_msgs = [
        {'role': 'system', 'content': system_prompt_rendered},
    ] + [{'role': m['role'], 'content': m['content']} for m in st.session_state.msgs]

    with st.chat_message('assistant'):
        # Process messages for ChatGroq
        messages_for_groq = [
            HumanMessage(content=msg['content']) if msg['role'] == 'user'
            else AIMessage(content=msg['content']) for msg in full_msgs
        ]

        # Call the RAG chain with user input
        response = rag_chain.invoke({"input": query})

        # Display the assistant's response
        st.markdown(response["answer"])

    # Append the assistant's response to the chat history
    st.session_state.msgs.append({'role': 'assistant', 'content': response["answer"]})
