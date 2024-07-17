#Chat with PS Deliverables .docx files
import umap
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

def get_vectorstore_from_pdf(uploaded_docs):
    
    st.write("Uploaded files are: \n")
    for file in uploaded_docs:
        st.write(file.name)
    
    with st.sidebar:
        with st.spinner('Loading the document...'):
            # document loading
            text = ""
            for file in uploaded_docs:
                if file.name.endswith('.pdf'):
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                elif file.name.endswith('.docx'):
                    # Create a temporary file and write the uploaded file's content to it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name
                    word_loader = Docx2txtLoader(tmp_file_path)
                    word_doc = word_loader.load()
                    text += word_doc[0].page_content

        st.success('Document loaded!', icon="✅")
    st.write(f'Loaded document:\n------------\n\n{text}')
    
    with st.sidebar:
        with st.spinner('Splitting the document into chunks...'):
            #document chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1028,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_text(text)
            document_chunks = text_splitter.create_documents(chunks)
        st.success(f'Document chunking completed! {len(chunks)} chunks', icon="✅")
    st.write(f'all {len(document_chunks)} chunks are following -\n\n')
    texts = [chunk.page_content for chunk in document_chunks]
    for i in range(len(texts)):
        st.write(f'chunk number {i}:\n------------\n\n{texts[i]}')

    with st.sidebar:
        with st.spinner('Creating embeddings of the overall text chunks...'):
            embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest",show_progress=True)
            global_embeddings = [embedding_model.embed_query(txt) for txt in texts]
        st.success(f"Total {len(global_embeddings)} embeddings created! and each embedding dimension is {len(global_embeddings[0])}", icon="✅")
            
        #Using UMAP algorithm now to reduce the dim of embeddings from 768 to 2
        dim = 2
        global_embeddings_reduced = reduce_cluster_embeddings(global_embeddings, dim)
        st.success("Reduced the dim of embeddings from 768 to 2 using UMAP!", icon="✅")
        print(f"len(global_embeddings_reduced): {len(global_embeddings_reduced)}")#22, coz 22 chunks so 22 embeddings
        print(f"global_embeddings_reduced[0]: {global_embeddings_reduced[0]}") #len will be 2 coz original 768 dim have been reduced to 2 dim using umap

    st.write("Plot the dimensionality reduced (2-dim) global_embeddings now to view the clusters")
    fig1 = plt.figure(figsize=(10, 8))
    plt.scatter(global_embeddings_reduced[:, 0], global_embeddings_reduced[:, 1], alpha=0.5)
    plt.title("Global Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(fig1) #plt.show()

    with st.sidebar:
        with st.spinner('Finding the optimal number of clusters in global_embeddings_reduced using BIC Score...'):
            labels, n_cluster = gmm_clustering(global_embeddings_reduced, threshold=0.5)

            plot_labels = np.array([label[0] if len(label) > 0 else -1 for label in labels])
            fig2 = plt.figure(figsize=(15, 12))
            unique_labels = np.unique(plot_labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                mask = plot_labels == label
                plt.scatter(global_embeddings_reduced[mask, 0], global_embeddings_reduced[mask, 1], color=color, label=f'Cluster {label}', alpha=0.5)
        st.success(f"GMM for clustering + optimal number of clusters using BIC completed! {n_cluster} clusters", icon="✅")
        plt.title("Cluster Visualization of Global Embeddings in 2D")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
    st.pyplot(fig2)


    with st.sidebar:
        simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]
        simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]
        df = pd.DataFrame({
            'Text': texts,
            'Embedding': list(global_embeddings_reduced),
            'Cluster': simple_labels
        })
        print(df)
        st.success("Built a df for storing Text chunks, Embeddings, and assigned Clusters", icon="✅")

        with st.spinner('Concatenating texts within each cluster...'):
            print("Concatenating texts within a cluster")
            clustered_texts = format_cluster_texts(df)
            print("Concatenation complete")
            st.success('Concatenating texts within each cluster complete', icon="✅")
        
    st.write(f'Concatenated texts within each cluster i.e. clustered_texts:\n')
    for cluster_number, cluster_text in clustered_texts.items():
            st.write(f'clustered_texts for Cluster {cluster_number}:\n------------------------------\n\n{cluster_text}')

    # Summarize texts within each cluster
    template = """You are an assistant to create a detailed summary of the text input prodived. Do not generate any preamble. 
    Text:
    {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="mistral:latest", temperature=0)
    chain = prompt | llm | StrOutputParser()

    summaries = {}
    with st.sidebar:
        with st.spinner('Summarize texts within each cluster...'):
            for cluster, text in clustered_texts.items():
                summary = chain.invoke({"input": text})
                summaries[cluster] = summary
        st.success('Summarization complete for texts within each cluster', icon="✅")
    st.write(f'Summaries within each cluster:\n------------\n\n')
    for cluster, summary in summaries.items():
        st.write(f'Summary of cluster number {cluster}:\n------------------------------\n\n{summary}')

    # Embed the cluster summaries now -> cluster those embeddings again -> Take cluster summaries again
    with st.sidebar:
        with st.spinner('Embed the cluster summaries now -> cluster those embeddings again -> Take cluster summaries again...'):
            embedded_summaries = [embedding_model.embed_query(summary) for summary in summaries.values()] #embedding all 36 clusters ka summaries now, so 36 embeddings
            embedded_summaries_np = np.array(embedded_summaries) #convert those 36 embeddings to np array
            labels, _ = gmm_clustering(embedded_summaries_np, threshold=0.5) #cluster those 36 embeddings using GMM
            simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

        with st.spinner('Checking how many clusters now left...'):
            clustered_summaries = {}
            for i, label in enumerate(simple_labels):
                if label not in clustered_summaries:   
                    clustered_summaries[label] = []
                clustered_summaries[label].append(list(summaries.values())[i])

    st.write(f'clustered_summaries:\n------------\n\n{clustered_summaries}')
    st.write(f'length of final clustered summaries is:\n------------\n\n{len(list(clustered_summaries.keys()))}')

    with st.sidebar:
        with st.spinner('Down to the final top level cluster, generating its summary...'):
            final_summaries = {}
            if len(list(clustered_summaries.keys())) == 1:#means you're down to 1 cluster
                #use the llm to write further summary
                final_summaries = {}
                for cluster, texts in clustered_summaries.items():
                    combined_text = ' '.join(texts)
                    summary = chain.invoke({"input": combined_text})
                    final_summaries[cluster] = summary
        st.success('Top level summary generated!', icon="✅")
    st.write(f'final_summaries:\n------------\n\n{final_summaries}') #this is now the highest level summary of our text

    texts_from_df = df['Text'].tolist() 
    texts_from_clustered_texts = list(clustered_texts.values())
    texts_from_final_summaries = list(final_summaries.values())
    combined_texts = texts_from_df + texts_from_clustered_texts + texts_from_final_summaries
    #combined_texts now contains the root documents and all sets of summaries
    
    # Now, use all combined_texts to build the vectorstore with Chroma, and store locally on disk
    persist_directory = "./local_embeddings/embeddings9"
    vectorstore = Chroma.from_texts(texts=combined_texts, embedding=embedding_model, persist_directory=persist_directory)
    # load the Chroma database from disk
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    with st.sidebar:
        st.success('Embeddings created and saved to vectorstore locally', icon="✅")

    return vectorstore

def reduce_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = max(2, int((len(embeddings) - 1) ** 0.5))
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

#1. Find the optimal number of clusters
def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234):
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1

#2. Fit the embeddings into the clusters using GMM as clustering algo
def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

#function to join or concatenate texts within a cluster
def format_cluster_texts(df):
    clustered_texts = {}
    for cluster in df['Cluster'].unique():
        cluster_texts = df[df['Cluster'] == cluster]['Text'].tolist()
        clustered_texts[cluster] = " --- ".join(cluster_texts) #join or concatenate texts within a cluster
    return clustered_texts

def get_conversational_retrieval_chain(history_aware_retriever):

    prompt_get_answer = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based only on the following context:\n\n{context} and Do not make up stuff you don't know."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    # llm = ChatOllama(model="mistral:latest", temperature=0)
    llm = ChatOllama(model="llama2:latest", temperature=0)
    document_chain = create_stuff_documents_chain(llm, prompt_get_answer)
    return create_retrieval_chain(history_aware_retriever, document_chain)


def get_history_aware_retriever(vector_store):
     
    prompt_search_query = ChatPromptTemplate.from_messages([
         MessagesPlaceholder(variable_name='chat_history'),
         "user", "{input}",
         "user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."
    ])
    # llm = ChatOllama(model="mistral:latest", temperature=0)
    llm = ChatOllama(model="llama2:latest", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    history_aware_retriever = create_history_aware_retriever(
         llm=llm,
         retriever=retriever,
         prompt=prompt_search_query
    )

    return history_aware_retriever


def get_response(user_input):
    history_aware_retriever = get_history_aware_retriever(st.session_state.vector_store)
    conversational_retrieval_chain = get_conversational_retrieval_chain(history_aware_retriever) #to actually answer the user question
    response = conversational_retrieval_chain.invoke({
         "chat_history": st.session_state.chat_history,
         "input": user_input
    })
    with st.sidebar:
         st.write(response)
    return response['answer']


# app config
st.set_page_config(page_title="Fully Local RAPTOR App", page_icon="")
st.title("Fully Local RAPTOR App")

with st.sidebar:
    st.subheader("Your documents")
    uploaded_docs = st.file_uploader(
        "Upload your files here and click on Process. Allowed extensions: .pdf, .docx", 
        type=(["pdf",".docx"]), 
        accept_multiple_files=True)
    process_button = st.button("Process")

if uploaded_docs == []:
    st.info("Please upload your files, then click on Process")
    print("uploaded_docs currently is empty: ", uploaded_docs)

else:
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am your Smart Project Assistant. How can I help you?")
            ]   
    if "conversation" not in st.session_state:
            st.session_state.conversation = None

    if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
    
    #track if st.button("Process") is clicked
    if "button_clicked" not in st.session_state:
         st.session_state.button_clicked = 0

    if process_button: 
        st.session_state.button_clicked = 1
        print("button clicked!")
        #build the vectorstore from uploaded_docs
        st.session_state.vector_store = get_vectorstore_from_pdf(uploaded_docs)

    if st.session_state.button_clicked == 1:
        #user input
        user_query = st.chat_input("Type your message here...")
        print(f"user_query: {user_query}")
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            response = get_response(user_query)
            st.session_state.chat_history.append(AIMessage(content=response))
            with st.sidebar:
                 st.subheader("st.session_state.chat_history")
                 st.write(st.session_state.chat_history)

        # show the HumanMessage and AIMessage as conversation on the webpage
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
