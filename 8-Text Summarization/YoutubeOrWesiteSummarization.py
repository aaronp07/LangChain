import validators, streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From Youtube or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From Youtube or Website")
st.subheader("Summarize URL")

# Get the Groq API Key and URL(Youtube or Website) to be summarize
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    
generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

prompt_template = """
Provide a summary of the following content in 300 words
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

if st.button("Summarize the Content from Youtube or Website"):
    # Validate all the input
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may a Youtube or Website Url")
    else:
        try:
            with st.spinner("Waiting..."):
                # Loading the Youtube or Website url
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                docs = loader.load()
                
                # Chain for summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                
                st.success(output_summary)
        except Exception as e:
            st.error(f"Exception: {e}")
                