import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain, LLMChain
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Set up the Streamlit App
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🧮")
st.title("Text 2 Math Problem Solver using Google Gemma 2")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API Key")
    st.stop
    
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Initialization the Wikipedia API Wrapper tool
wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name = "Wikipedia",
    func = wikipedia_wrapper.run,
    description = "A tool for searching the internet to find the various information on the topic mentioned."
)

# Initialize the LLM Math Chain Tool
math_chain = LLMMathChain.from_llm(llm=llm)

calculator_tool = Tool(
    name = "Calculator",
    func = math_chain.run,
    description = "A tool for answering math related question. Only input mathematical expression need to be provided"
)

# Prompt
prompt = """
Your a agent tasked for solving users mathematical question. Locally arrive at the solution and provide a detailed explaination
and display it point wise for the question below
Question: {question}
Answer:
"""

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools [Wikipedia, Calculator and Reasoning] into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name = "Reasoning Tool",
    func = chain.run,
    description = "A tool for answering logic-based and reasoning questions."
)

# Initialize the Agents
assistant_agent = initialize_agent(
    tools = [wikipedia_tool, calculator_tool, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors = True
)

# Function to generate the response
def generate_response(question):
    response = assistant_agent.invoke({'input': question})
    return response

# Check the messages to store previous conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math Chatbot who can answer all your maths questions"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
# Let's start the interaction
question = st.text_area("Enter your question", "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_callback])
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter your question")