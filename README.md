**************** Complete LangChain ************************
* Step 1:
    To read the Document Loaders install - "langchain_community" 

# 1.1 OpenAI
    1. ChatPromptTemplate
        * Installation:-
            a) langchain-openai
            b) langchain
            c) python-dotenv

        * Code:-
            import os
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from dotenv import load_dotenv
            load_dotenv()

            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
            # LangSmith Tracking
            os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
            os.environ['LANGSMITH_TRACING'] = "true"
            os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

            # Large Language Model - GPT-4o
            llm = ChatOpenAI(model='gpt-4o')
            
            # Input and get response from LLM
            result = llm.invoke("what is generative ai?")
            print(result)

            # Chat Prompt Template
            prompt = ChatPromptTemplate([
                ("system", "You are an expert AI Engineer. Provide me answers based on the question"),
                ("user", "{input}")
            ])

            # Chain of prompt
            chain = prompt | llm
            response = chain.invoke({"input": "Can you tell me about LangSmith?"})
            print(response)

            # StrOutput Parser
            from langchain_core.output_parsers import StrOutputParser

            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser

            response = chain.invoke({"input": "Can you tell me about LangSmith?"})
            print(response)

    2. WebScrapeChat
        * Installation:-
            a) langchain-openai
            b) langchain
            c) langchain_community
            d) bs4
            e) langchain-text-splitters
            f) faiss-cpu
            g) python-dotenv

        * Code:-
            import os
            from langchain_openai import ChatOpenAI
            from langchain_community.document_loaders import WebBaseLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.documents import Document
            from langchain.chains import create_retrieval_chain
            from dotenv import load_dotenv
            load_dotenv()

            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
            # LangSmith Tracking
            os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
            os.environ['LANGSMITH_TRACING'] = "true"
            os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

            # Step 1: Data Ingestion - From the website we need to scrape the data
            loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
            docs = loader.load()

            # Step 2: Text into Chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(docs)

            # Step 3: Embedding Techniques
            embeddings = OpenAIEmbeddings()

            # Step 4: Vector Stores DB
            vectordb = FAISS.from_documents(documents, embeddings)

            # Step 5: Query from a Vector Store DB
            query = "LangSmith has two usage limits: total traces and extended"
            result = vectordb.similarity_search(query)

            llm = ChatOpenAI(model='gpt-4o')

            # Step 6: Chat Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """
                Answer the following question based on the provided context:
                <context>{context}</context>
                """
            )

            # Step 7: Document Chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            document_chain.invoke({
                "input": "LangSmith has two usage limits: total traces and extended",
                "context": [Document(page_content="LangSmith has two usage limits: total traces and extended traces. These correspond to the two metrics we've been tracking on our usage graph.")]
            })

            # Step 8: Retrieval Chain
            retriever = vectordb.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Step 9: Get the respose from the LLM
            response = retrieval_chain.invoke({"input": "LangSmith has two usage limits: total traces and extended"})
            response['answer']

# 1.2 Ollama
    1. General Chat using 'gemma:b2' - Google Model
        * Installation:-
            a) langchain-ollama
            b) langchain
            c) langchain_community
            d) streamlit
            e) python-dotenv

        * Code:-
            import os
            from dotenv import load_dotenv
            from langchain_community.llms import Ollama
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            import streamlit as st

            load_dotenv()

            # Step 1: LangSmith Tracking
            os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
            os.environ['LANGSMITH_TRACING'] = "true"
            os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

            # Step 2: Prompt Template
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant. Please respond to the question asked"),
                    ("user", "Question:{question}")
                ]
            )

            # Step 3: Streamlit Framework
            st.title("LangChain Demo with Gemma Model")
            input_text = st.text_input("What question you have in mind?")

            # Step 4: Ollama Llama2 model
            llm = Ollama(model="gemma:2b")
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser

            if input_text:
                st.write(chain.invoke({"question": input_text}))

# 1.3 Data Ingestion - Reading File (Text file, PDF, Web Content, Web PDF File, Wikipedia)
    1. Load Text File
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader('./data/speech.txt')
        text_documents = loader.load()

    2. Load PDF File - Install "pypdf"
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader('./data/yolov7.pdf')
        docs = loader.load()

    3. Load Web based content - Install "bs4" # Beautiful Soup
        from langchain_community.document_loaders import WebBaseLoader
        import bs4 # Beautiful Soup is to retrieve data from html attribute

        loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                            bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                                class_=("post-title", "post-content", "post-header")
                            )))
        loader.load()

    4. Web PDF file read - Install "arxiv" and "pymupdf"
        from langchain_community.document_loaders import ArxivLoader

        docs = ArxivLoader(query="Attention is All you Need", load_max_docs=2).load()
        print(docs)

    5. Wikipedia content read - Install "wikipedia"
        from langchain_community.document_loaders import WikipediaLoader

        docs = WikipediaLoader(query="Generative AI", load_max_docs=2).load()
        print(docs)
        
# 1.4 Text Chunks (Data Transformer) - Reading File, Split the text into chunks
    1. Load the PDF file
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader('../3.2-Data Ingestion/data/yolov7.pdf')
        docs = loader.load()

    2. Split the Text - Install "langchain-text-splitters"
        a) RecursiveCharacterTextSplitter - (Paragraphs → Sentences → Words → Characters)
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            final_documents = text_split.split_documents(docs)

            print(final_documents[0].page_content)
            print('\n#########################################################\n')
            print(final_documents[1].page_content)

        b) CharacterTextSplitter - (Splits text based on a fixed character length. It uses no understanding of structure)
            from langchain_text_splitters import CharacterTextSplitter

            text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
            text_splitter.split_documents(docs)

    ## Text Chunks (Data Transformer) - Reading HTML and Text Splitter
        from langchain_text_splitters import HTMLHeaderTextSplitter

        url = "https://artificialanalysis.ai/"

        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]

        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
        html_header_splits = html_splitter.split_text_from_url(url)
        html_header_splits

    ## Text Chunks (Data Transformer) - Reading JSON Data
        1. Get the Json API data
            import json
            import requests

            json_data = requests.get('https://api.smith.langchain.com/openapi.json').json()

        2. Split the Text using RecursiveJsonSplitter
            from langchain_text_splitters import RecursiveJsonSplitter

            json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
            json_chunks = json_splitter.split_json(json_data)

            json_chunks

        3. The splitter can also output as Document
            docs = json_splitter.create_documents(texts=[json_data])

            for doc in docs[:3]:
                print(doc)

        4. Text Split
            texts = json_splitter.split_text(json_data)
            print(texts[0])
            print(texts[1])

# 1.5 Embeddings Techniques
    * Get Environment Details
        i. Install "python-dotenv"
        ii. Get the value from .env
            import os
            from dotenv import load_dotenv
            load_dotenv()

            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    * ChromaDb Install - "chromadb"

    1. OpenAI - Install "langchain-openai"
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        text = "This is a tutorial on OPENAI embedding"
        query_result = embeddings.embed_query(text)
        len(query_result) # Check the dimension

        # Fixed Dimension
        embeddings_1024 = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1024)
        text = "This is a tutorial on OPENAI embedding"
        query_result = embeddings_1024.embed_query(text)

        * Read the text file, Split, Embedding, Store Chroma Db, Retrieve data from query
            # 1. Split using RecursiveCharacterTextSplitter
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            final_doc = text_splitters.split_documents(docs) 

            # 2. Embedding
            embeddings_1024 = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1024)

            # 3. Vector Embedding and Vector StoreDB
            from langchain_community.vectorstores import Chroma

            db = Chroma.from_documents(final_doc, embeddings_1024)
            
            # 4. Retrieving the data based on the query
            query = "Kalam collapsed and died from an apparent cardiac arrest"
            retrieved_results = db.similarity_search(query)
            print(retrieved_results)

    2. Ollama - Install "langchain-ollama"
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model="gemma:2b")  # By default it uses llama2
        r1 = embeddings.embed_documents([
            "Alpha is the first letter of Greek alphabet",
            "Beta is the second letter of Greek alphabet"
        ])

        len(r1[0]) # Check the dimension - 2048

        embeddings.embed_query("What is the second letter of Greek alphabet")
        # Other Embedding Models
        # https://ollama.com/blog/embedding-models

        embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        text = "This is a test document."
        query_result = embeddings.embed_query(text)
        len(query_result) # Check the dimension - 1024

    3. Hugging Face - Install "sentence_transformers" and "langchain_huggingface"
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        text = "this is atleast documents"
        query_result = embeddings.embed_query(text)
        len(query_result) # Check the dimension - 384

        # Documents Embedding
        doc_result = embeddings.embed_documents([
            text,
            "This is not a test documents."
        ])
        len(doc_result[0])

# 1.6 Vector Stores - https://python.langchain.com/v0.2/docs/integrations/vectorstores/
    1. FAISS (Facebook AI Similarity Search) - Install "faiss-cpu"
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS

        # Step 1: Load the Document
        loader = TextLoader('./data/speech.txt')
        documents = loader.load()

        # Step 2: Splitter the text
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)

        # Step 3: Create Vector Store
        embeddings = OllamaEmbeddings(model="gemma:2b")
        db = FAISS.from_documents(docs, embeddings)

        # Step 4: Querying
        query = "Kalam collapsed and died from an apparent cardiac arrest?"
        docs = db.similarity_search(query)
        docs[0].page_content

        # Step 5: Save in local
        db.save_local('faiss_index')

        # Step 6: Load from local
        new_db = FAISS.load_local('./faiss_index', embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(query)
        print(docs[0].page_content)

        # Step 7: Similarity Seach with Score
        doc_and_score = db.similarity_search_with_score(query)
        doc_and_score

        # Step 8: Retriever option
        retriever = db.as_retriever()
        docs = retriever.invoke(query)
        docs[0].page_content        

    2. Chroma DB - Install "langchain_chroma" and "chromadb"
        # Building a sample vectordb
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma

        # Step 1: Load the Document
        loader = TextLoader('./data/speech.txt')
        documents = loader.load()

        # Step 2: Splitter the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)

        # Step 3: Create Vector Store
        embeddings = OllamaEmbeddings(model="gemma:2b")
        vectordb = Chroma.from_documents(docs, embeddings)

        # Step 4: Querying
        query = "Kalam collapsed and died from an apparent cardiac arrest?"
        docs = vectordb.similarity_search(query)
        docs[0].page_content

        # Step 5: Save in local
        vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='./chroma_db')

        # Step 6: Load from local
        db2 = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
        docs = db2.similarity_search(query)
        print(docs[0].page_content)

        # Step 7: Similarity Seach with Score
        docs = vectordb.similarity_search_with_score(query)
        docs

        # Step 8: Retriever option
        retriever = vectordb.as_retriever()
        retriever.invoke(query)[0].page_content

------------------------------------------------------------------------------------------------------------------------------------

# 2.1 Basics Of LangChain
    1. Groq API - LangChain Expression Language (Open Source Model) - Install "langchain_groq" and "langchain_core"
        # OpenAI API Key and Open Source Models [Llama3 (Meta), Gemma2 (Google), Mistral] Platform use Groq API Key
        import os
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from dotenv import load_dotenv
        load_dotenv()

        groq_api_key = os.getenv('GROQ_API_KEY')
        model = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)

        messages = [
            SystemMessage(content = "Translate the following question from English to French language"),
            HumanMessage(content = "Hello!, How are you?")
        ]

        result = model.invoke(messages)        

        parser=StrOutputParser()
        parser.invoke(result)

        # using LCEL - Chain the components
        chain = model | parser
        chain.invoke(messages)

        # Prompt Templates
        generic_template = "Translate the following into {language}:"

        prompt = ChatPromptTemplate.from_messages([("system", generic_template),("user", "{text}")])
        result = prompt.invoke({"language":"French", "text":"Hello"})
        result.to_messages()

        # Chaining together components with LCEL
        chain = prompt | model | parser
        chain.invoke({"language":"French", "text":"Hello"})

    2. FastAPI (Serve.py)
        * Installation:-
            langchain>=0.3.60
            langchain-core>=0.3.60
            langsmith==0.0.99
            langchain-community==0.0.21
            langchain-openai==0.0.8
            langchain-groq>=0.3.0
            langchain-ollama>=0.3.3
            langchain-huggingface==0.2.0
            langchain-chroma==0.2.4
            langchain-text-splitters==0.0.1
            fastapi>=0.105
            pydantic==1.10.13
            uvicorn
            python-dotenv
            sse-starlette==1.6.5

        * Code:-
            from fastapi import FastAPI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_groq import ChatGroq
            from langserve import add_routes
            import os
            from dotenv import load_dotenv
            load_dotenv()

            # Step 1: Set the model
            groq_api_key = os.getenv('GROQ_API_KEY')
            model = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)

            # Step 2: Create prompt template
            system_template = "Translate the following into {language}:"
            prompt_template = ChatPromptTemplate([('system', system_template), ('user', '{text}')])

            # Step 3: StrOutputParser
            parser = StrOutputParser()

            # Step 4: Create chain
            chain = prompt_template | model | parser

            # App Definition
            app = FastAPI(title="LangChain Server", version="1.0", description="A Simple API server using LangChain and LangServe")

            # Adding Chain Routes
            add_routes(app, chain, path="/chain")

            if __name__ == "__main__":
                import uvicorn
                uvicorn.run(app, host="127.0.0.1", port=8000)

------------------------------------------------------------------------------------------------------------------------------------

# 3.1 Build Chatbot with Message History
    * Installation:-
        langchain==0.3.26
        langchain-core==0.3.66
        langchain-text-splitters==0.3.8
        langsmith==0.4.1
        pydantic==2.11.7
        pydantic-settings==2.10.0
        langchain-community==0.3.26  

    * Code:-
        import os
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        from langchain_core.messages import AIMessage
        from langchain_community.chat_message_histories import ChatMessageHistory
        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.runnables import RunnableWithMessageHistory
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
        from operator import itemgetter
        from langchain_core.runnables import RunnablePassthrough

        from dotenv import load_dotenv
        load_dotenv()

        groq_api_key = os.getenv('GROQ_API_KEY')

        model = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)
        model.invoke([HumanMessage(content="Hi, My name is Aaron and I'm senior developer in asp.net and learning AI engineer")])

        model.invoke(
            [
                HumanMessage(content="Hi, My name is Aaron and I'm senior developer in asp.net and learning AI engineer"),
                AIMessage(content="Hi Aaron,\n\nIt's great to meet you!\n\nThat's fantastic that you're transitioning from ASP.NET development into the world of AI engineering. Your background in software development will be incredibly valuable as you learn AI. \n\nWhat aspects of AI are you most interested in exploring? \n\nDo you have any specific projects or goals in mind?\n\nI'm here to help you along your journey.  I can provide information, answer questions, and even help you brainstorm ideas.  Let me know how I can be of assistance!\n"),
                HumanMessage(content="Hey, What's my name, previous work and currently what am doing?")
            ]
        )

        # Message History
        store = {} # Created Dict

        def get_session_history(session_id: str)->BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        with_message_history = RunnableWithMessageHistory(model, get_session_history)
        
        # chat1 session id
        config = {"configurable": {"session_id": "chat1"}}
        
        response = with_message_history.invoke(
            [HumanMessage(content="Hi, My name is Aaron and I'm senior developer in asp.net and learning AI engineer")],
            config=config
        )
        response.content

        with_message_history.invoke(
            [HumanMessage(content="What is my name?")],
            config=config
        )

        # Change the config --> Session id - chat2 session id
        config1 = {"configurable": {"session_id": "chat2"}}
        response = with_message_history.invoke([HumanMessage(content="What's my name?")], config=config1)
        response.content

        # Prompt Template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Answer all the question to the best of your ability"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        chain = prompt | model
        chain.invoke({"messages":[HumanMessage(content="Hi my name is Aaron")]})

        with_message_history = RunnableWithMessageHistory(chain, get_session_history)

        # chat3 session id
        config = {"configurable": {"session_id": "chat3"}}
        response = with_message_history.invoke(
            [HumanMessage(content="Hey my name is Aaron")],
            config=config
        )
        response.content

        # Multiple Input Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Answer all the questions to the best of your ability in {language}"),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        chain = prompt | model

        response = chain.invoke({"messages": [HumanMessage(content="Hi my name is Aaron")], "language":"tamil"})
        response.content

        with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")
        
        # chat4 session id
        config = {"configurable": {"session_id": "chat4"}}

        response = with_message_history.invoke({"messages": [HumanMessage(content="Hi, Im Aaron")], "language":"tamil"}, config=config)
        response.content

        # Managing the Conversation History
        trimmer = trim_messages(
            max_tokens=45,
            token_counter=model,
            strategy="last",
            allow_partial=False,
            start_on="human",
            include_system=True
        )

        messages = [
            SystemMessage(content="You're a good assistant"),
            HumanMessage(content="Hi! I'm bob"),
            AIMessage(content="Hi!"),
            HumanMessage(content="I like vanilla ice cream"),
            AIMessage(content="nice"),
            HumanMessage(content="whats 2 + 2"),
            AIMessage(content="4"),
            HumanMessage(content="thanks"),
            AIMessage(content="no problem!"),
            HumanMessage(content="having fun?"),
            AIMessage(content="yes!"),
        ]

        trimmer.invoke(messages)

        chain = (RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) | prompt | model)
        response = chain.invoke(
            {
                "messages": messages + [HumanMessage(content="what ice cream do i like?")],
                "language": "English"
            }
        )

        response.content

        response = chain.invoke({
            "messages": messages + [HumanMessage(content="What math problem did i ask?")],
            "language": "English"
        })

        print(response.content)

        # Lets wrap this in the Message History
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="messages"
        )

        config = {"configurable":{"session_id": "chat6"}}

        response = with_message_history.invoke(
            {
                "messages": messages + [HumanMessage(content="what's my name?")],
                "language" : "English"
            },
            config=config
        )

        response.content

# 4.1 Vector Stores and Retrievers
    * Installation:-
        langchain==0.3.26
        langchain-groq==0.3.4
        langchain-chroma==0.1.4
        langchain-huggingface==0.2.0
        pydantic==2.11.7

    * Code:-
        import os
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_chroma import Chroma
        from langchain_core.runnables import RunnableLambda
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough

        groq_api_key = os.getenv('GROQ_API_KEY')
        os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

        llm = ChatGroq(groq_api_key=groq_api_key, model='llama3-8b-8192')

        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # Sentence 

        # Load the document
        documents = [
            Document(
                page_content="Dogs are great companions, known for their loyalty and friendliness.",
                metadata={"source": "mammal-pets-doc"}
            ),
            Document(
                page_content="Cats are independent pets that often enjoy their own space.",
                metadata={"source": "mammal-pets-doc"}
            ),
            Document(
                page_content="Gold fish are popular pets for beginners, requiring relatively simple care.",
                metadata={"source": "fish-pets-doc"}
            ),
            Document(
                page_content="Parrots are intelligent birds capable of mimicking human speech.",
                metadata={"source": "bird-pets-doc"}
            ),
            Document(
                page_content="Rabbits are social animals that need plenty of space to hop around.",
                metadata={"source": "mammal-pets-doc"}
            )
        ]

        # Vector Store
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

        # Await
        await vectorstore.asimilarity_search('cat')

        # Runnable Lambda - Retriever Technique-1
        retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1) # k top value
        retriever.batch(['cat', 'dog'])

        # Vectore Store as_retriever - Technique-2
        retriver_vectore = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":1}
        )

        retriver_vectore.batch(['cat', 'dog'])

        # Sample Chat
        message = """
            Answer this question using the provided context only.
            
            {question}
            
            Context:
            {context}
        """

        prompt = ChatPromptTemplate.from_messages([('human', message)])

        rag_chain = {"context": retriver_vectore, "question": RunnablePassthrough()} | prompt | llm

        response = rag_chain.invoke('tell me about rat')
        print(response.content)

# 5.1 Conversation Q&A Chatbot
    * Installation:-
        langchain==0.3.26
        langchain-chroma==0.1.4
        langchain-community==0.3.26
        langchain-core==0.3.66
        langchain-groq==0.3.4
        langchain-huggingface==0.2.0
        langchain-text-splitters==0.3.8
        bs4==0.0.2

    * Code:-
        import os
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        import bs4
        from langchain.chains import create_history_aware_retriever
        from langchain_core.prompts import MessagesPlaceholder
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_community.chat_message_histories import ChatMessageHistory
        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory

        groq_api_key = os.getenv('GROQ_API_KEY')

        llm = ChatGroq(groq_api_key=groq_api_key, model='llama3-8b-8192')

        os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        # Step 1: Load the HTML content
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(class_=("post-content", "post-header"))
            )
        )

        docs = loader.load()

        # Step 2: Split the text into chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Step 3: Store the vectore in Chroma db
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Step 4: Prompt Template
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )

        # Step 5: Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Step 6: Without History Response
        response = rag_chain.invoke({"input": "What is Self-Reflection"})
        response['answer']

        contextualize_que_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_que_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_que_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_que_prompt)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Step 8: Reponse
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chat_history = []
        question = "What is Self-Reflection"
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})

        chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=response['answer'])
            ]
        )

        response['answer']

        question2 = "Tell me more about it?"
        response2 = rag_chain.invoke({"input": question, "chat_history": chat_history})
        response2['answer']

        # Chat Session Id
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        conversational_rag_chain.invoke(
            {"input": "What is task decomposition?"},
            config={
                "configurable": {"session_id": "abc123"}
            },
        )["answer"]

        conversational_rag_chain.invoke(
            {"input": "What are common ways of doing it?"},
            config={
                "configurable": {"session_id": "abc123"}
            },
        )["answer"]


