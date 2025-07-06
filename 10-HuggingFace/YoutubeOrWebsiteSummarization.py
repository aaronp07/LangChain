import validators, streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Set up Streamlit App
st.set_page_config(page_title="HuggingFace: Summarize Text from Youtube or Website", page_icon="🦜")
st.title("🦜HuggingFace: Summarize Text from Youtube or Website")
st.subheader("Summarize URL")

# Get the HuggingFace API Token
with st.sidebar:
    hf_api_key = st.text_input("HuggingFace API Token", type="password")
    
generic_url = st.text_input("URL", label_visibility="collapsed")

# Google Gemma Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b",
    huggingfacehub_api_token=hf_api_key,
    temperature=0.7,
    max_new_tokens=150,
    task="conversational"  # ✅ this fixes the error
)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""

# Prompt
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

if st.button("Summarize the Content from Youtube or Website"):
    # Validate all the inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may be a Youtube video or Website Url")
    else:
        try:
            with st.spinner("Waiting..."):
                # Loading the Youtube or Website Url
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                docs = loader.load()
                
                #Chain for summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.invoke(docs)
                
                st.success(output_summary)
        except Exception as e:
            st.error(f"Exception: {e}")