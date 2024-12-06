import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HUGGING_API_KEY'] = os.getenv("HUGGING_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Load environment variables (if needed for API keys)

# Set up the title and layout
# Set up the title and layout
st.set_page_config(page_title="Pratham Chawla Portfolio", page_icon=":guardsman:", layout="wide")

# Title Section
st.markdown(
    """
    <div style="text-align: center; font-size: 40px; font-weight: bold;">Pratham Chawla</div>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; font-size: 20px;">Ask me anything about my work, projects, or experience.</div>
    """, 
    unsafe_allow_html=True
)


# Left sidebar with skills and achievements
with st.sidebar:
    st.header("Skills")

    st.subheader("Languages")
    st.write("Python, SQL (Structured Query Language)")

    st.subheader("Data Science")
    st.write("Data science pipeline (cleansing, wrangling, visualization, modeling, interpretation), Statistics, Business Intelligence Tools (SSIS, SSMS, Power-BI)")

    st.subheader("Computer Vision")
    st.write("Object Detection and Tracking, Image Processing")
    
    st.header("Gen AI")
    st.write("Document Chains, History-Aware Retrieval, Prompt Engineering")
    
    st.subheader("Soft Skills")
    st.write("Project Management, Teamwork, Algorithms")

# Streamlit setup for the user interface


loader = TextLoader("About Pratham.txt")  # Default text loader
text_documents = loader.load()

# Split the documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(text_documents)

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant specialized in answering questions about Pratham Chawla, his projects, skills and Experience. "
    "When answering, provide detailed and structured answers. If the user asks for Pratham's projects,skills and Experience "
    "list them with names, descriptions, and any related achievements. "
    "If the question is about a specific project, give an elaborative response with relevant details. "
    "If any technical terms are mentioned, explain them in simple terms."
    "\n\n{context}"
)


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history,"
    "and give a elaborative answer according to question "
    "asked and if there are some technical terms explain them"    
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),
    ]
)

# Create history-aware retriever using Langchain
history_aware_retriver = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),
    ]
)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriver,question_answer_chain)

# Streamlit input form for user question
user_input = st.text_input("Ask a question:")

# When the user submits a question
if user_input:
    response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
    
    # Store the question and answer in chat history
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"])
        ]
    )

    # Display the answer in Streamlit
    st.write("Answer:", response["answer"])

    

# Optionally, you can also provide a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
    st.write("Chat history cleared.")

def download_chat_history():
    chat_text = ""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            chat_text += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            chat_text += f"AI: {message.content}\n"
    return chat_text

# Add a download button for the chat history
st.download_button(
    label="Download Chat History",
    data=download_chat_history(),
    file_name="chat_history.txt",
    mime="text/plain"
)

# Display Deployed Projects below the chatbot
st.markdown(
    """
    <div style="text-align: center; font-size: 30px; font-weight: bold;">Deployed Projects</div>
    """, 
    unsafe_allow_html=True
)

# Google Deep Dream and Neural Style Transfer
st.markdown(
    """
    **1. Google Deep Dream and Neural Style Transfer**: 
    Implemented Google Deep Dream for surreal image transformations and applied Neural Style Transfer to blend artistic styles. 
    [View Project](https://deepdream-nst.streamlit.app/)
    """, 
    unsafe_allow_html=True
)

# Image Captioning on Flickr 8K Dataset
st.markdown(
    """
    **2. Image Captioning on Flickr 8K Dataset**: 
    Developed an image captioning system using CNN+RNN for generating captions from the Flickr 8K dataset.
    [View Project](https://image-captioningcnnrnn.streamlit.app/)
    """, 
    unsafe_allow_html=True
)

# Next Words Prediction Using LSTM vs GRU
st.markdown(
    """
    **3. Next Words Prediction Using LSTM vs GRU**: 
    Created models using LSTM and GRU for predicting the next word, showcasing efficiency and accuracy.
    [View Project](https://next-words-prediction.streamlit.app/)
    """, 
    unsafe_allow_html=True
)

# Point Cloud Classification
st.markdown(
    """
    **4. Point Cloud Classification**: 
    Applied deep learning for classifying 3D point cloud data into categories like buildings, vehicles, and trees.
    [View Project](https://point-cloud-classification-s.streamlit.app/)
    """, 
    unsafe_allow_html=True
)

# Face Recognition with Siamese Networks
st.markdown(
    """
    **5. Face Recognition with Siamese Networks**: 
    Built a face recognition system using Siamese Networks for one-shot learning and minimal data.
    [View Project](https://siamese-network.streamlit.app/)
    """, 
    unsafe_allow_html=True
)

# Add a heading for Other Projects
st.markdown(
    """
    <div style="text-align: center; font-size: 30px; font-weight: bold;">Other Projects</div>
    """, 
    unsafe_allow_html=True
)

# List of Other Projects with concise descriptions
st.markdown(
    """
    **1. DeepFace vs CNN+RNN Approach for Emotion Detection**: 
    Comparative study between DeepFace and CNN+RNN for emotion detection, showcasing superior results with CNN+RNN for sequential facial expressions.

    **2. Applications of YOLO**:
    - **Person Loitering Detection**: Identified unusual lingering behavior.
    - **Abandoned Object Detection**: Detected unattended objects in public spaces.
    - **Vehicle Analytics**: Performed vehicle counting, wrong-direction detection, and lane violation monitoring.
    - **Parking Space Availability**: Real-time detection of available and occupied parking spaces.

    **3. Video Title Generation Using CNN+RNN**: 
    Generated video titles by integrating CNNs for frame features and RNNs for text generation, improving video accessibility.

    **4. Image Search with Text Queries on Flickr 8K**: 
    Built a cross-modal retrieval system enabling image searches using textual queries, aligning visual and textual semantics.
    
    **5. Point Cloud Classification**: 
    Used deep learning to classify 3D point cloud data into categories like buildings and vehicles.

    **6. 3D Point Cloud and Mesh Generation Using a Single Image**: 
    Reconstructed 3D point clouds and meshes from single 2D images for applications in 3D modeling, AR, and VR.

    **7. Web Page Q&A App Using Generative AI**: 
    Built an AI-powered app to answer questions based on webpage content, enhancing information retrieval and user interaction.
    """, 
    unsafe_allow_html=True
)


