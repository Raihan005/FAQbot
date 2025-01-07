import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

template = """
You are a helpful AI assistant that specializes in providing FAQ support for our company.
Context: You should always:
- Be concise and direct
- Respond in a friendly, professional tone
- Provide specific examples when relevant
- If you don't know something, say so directly
Current conversation:
{history}
Human: {input}
Assistant:
"""

def initialize_session_state():
    if 'chain' not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.7
        )
        prompt = PromptTemplate(template=template)
        
        memory = ConversationBufferMemory()
        st.session_state.chain = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
def clear_chat():
    st.session_state.messages = []
    st.session_state.chain.memory.clear()
def main():
    st.set_page_config(layout='centered')
    st.title("FAQ Chatbot")

    with st.sidebar:
        st.title("Chat Controls")
        if st.button("Clear Chat"):
            clear_chat()
            
    # Initialize session state
    initialize_session_state()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask your question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("..."):
                response = st.session_state.chain.predict(input=prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
