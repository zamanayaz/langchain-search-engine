import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler


api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper= api_wrapper_arxiv)
search = DuckDuckGoSearchRun(name="Search")


st.title("Langchain - Search Engine with Wikipedia and Arxiv Tools.") 

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your groq api key.", type="password")


if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role":"assistant", "content":"Hi, I am a chatbot who can search the web. How can I help you? "},
    ]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])


if (prompt:=st.chat_input(placeholder="What is Data Science?")) and api_key:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    llm = ChatGroq(groq_api_key = api_key, model="gemma2-9b-it", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
else:
    st.warning("Please enter your api key.")