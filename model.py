import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_huggingface import ChatHuggingFace , HuggingFaceEmbeddings , HuggingFaceEndpoint
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import  AIMessage , HumanMessage
import streamlit as st



load_dotenv()

st.set_page_config(page_title="TubeMentor AI", page_icon="🎬")
st.title("🎬 TubeMentor AI")
st.markdown(
    "Paste a YouTube URL and chat with the video using AI."
)

#Session State
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

#Chat Model
@st.cache_resource
def load_models():
    llm=HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task='text-generation')
    model=ChatHuggingFace(llm=llm)

    #Embedding Model
    embed=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

    return model , embed

model, embed = load_models()
    
parser=StrOutputParser()



def extract_transcript(yt_url):

    parsed = urlparse(yt_url)

    if "youtu.be" in parsed.netloc:
        video_id = parsed.path[1:]
    else:
        video_id = yt_url.split("v=")[1].split("&")[0]

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id,languages=["hi", "en"])

    text = " ".join([item.text for item in transcript])

    return text

#Runnable1
transcript_loader = RunnableLambda(extract_transcript)



def text_split(text):
    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    return chunks
#Runnable2
spliter = RunnableLambda(text_split)



def create_vector_store(chunks: list[str]) -> Chroma:
    # Delete existing collection to avoid stale data from previous videos
    try:
        Chroma(
            persist_directory="./chroma_db",
            embedding_function=embed,
            collection_name="tube_mentor"
        ).delete_collection()
    except Exception:
        pass

    return Chroma.from_texts(
        texts=chunks,
        embedding=embed,
        persist_directory="./chroma_db",
        collection_name="tube_mentor"
    )

#Runnable3
vector_store_builder = RunnableLambda(create_vector_store)



def retrival(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k":20,
            "lambda_mult":0.4
                }
            )
    return retriever



prompt = ChatPromptTemplate.from_messages([
    
    (
        "system",
        """
You are TubeMentor AI, an assistant that answers questions about a YouTube video.

Rules:
1. Answer only from the provided context.
2. Do not use outside knowledge.
3. If the answer is not found in the context, say:
   "I could not find this information in the video."
4. Keep answers clear and concise.
"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        """
Context:
{context}

Question:
{question}
"""
    )
])



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

formatter = RunnableLambda(format_docs)

url = st.text_input(
    "Paste YouTube URL"
)

Knowledge_chain = transcript_loader | spliter  | vector_store_builder


if st.button("Process Video"):

    if not url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    else:
        with st.spinner("Processing video...."):

            try:

                db = Knowledge_chain.invoke(url)

                st.session_state["retriever"] = retrival(db)
                
                st.session_state["chat_history"] = []

                st.success("Video processed successfully!")

            except Exception as e:

                st.error(str(e))

for msg in st.session_state.get("chat_history", []):

    if isinstance(msg, HumanMessage):

        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):

        with st.chat_message("assistant"):
            st.write(msg.content)

if st.session_state["retriever"]:
    rag_chain = (
    {
        "context": st.session_state["retriever"] | formatter,
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(
            lambda _: st.session_state.get("chat_history", [])
        )
    }
    | prompt
    | model
    | parser
)




# Chat input
query = st.chat_input("Ask about the video. Type 'exit' to stop.")

if query:

    # Exit command
    if query.lower().strip() == "exit":

        st.session_state["retriever"] = None
        st.session_state["chat_history"] = []

        st.success("Session ended. Paste a new YouTube URL.")

        st.rerun()

    if st.session_state["retriever"] is None:
        st.warning("Please process a YouTube video first.")

    else:

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(query)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state["chat_history"].append(
            HumanMessage(content=query)
        )

        st.session_state["chat_history"].append(
            AIMessage(content=answer)
        )


# ---------------- Footer ----------------



st.markdown(
    """
    <div style="text-align:center; color:#9CA3AF; font-size:14px;">
        Built with ❤️ by <b>Rajhans Bagri</b><br>
        © 2026 All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)