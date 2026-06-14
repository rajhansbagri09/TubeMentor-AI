from langchain_huggingface import ChatHuggingFace , HuggingFaceEmbeddings , HuggingFaceEndpoint
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage
import streamlit as st



load_dotenv()

st.set_page_config(page_title="TubeMentor AI", page_icon="🎬")
st.title("🎬 TubeMentor AI")

#Chat Model
llm=HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task='text-generation')
model=ChatHuggingFace(llm=llm)

#Embedding Model
embed=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

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

def retirval(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k":20,
            "lambda_mult":0.4
                }
            )
    return retriever

#Runnable4
#

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

url=input("Please give your YouTube Video URL: ")

Knowledge_chain = transcript_loader | spliter  | vector_store_builder



try:
    db = Knowledge_chain.invoke(url)
except Exception as e:
    print(f"Error processing video: {e}")
    exit()
    
    
    

retriever_obj=retirval(db)

chat_history=[]

rag_chain = (
    {
        "context": retriever_obj | formatter,
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(lambda _: chat_history)
    }
    | prompt | model | parser
    
)


while True:
    query = input("\nAsk Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer = rag_chain.invoke(query)

    print("\nAnswer:")
    print(answer)
    chat_history.append(
    HumanMessage(content=query)
     )

    chat_history.append(
        AIMessage(content=answer)
    )