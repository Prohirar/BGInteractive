import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
#import datetime


load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") 


def get_conversational_chain():
    print("===========Start get_conversational_chain")
    prompt_template = """
    You are helpful and informative bot that answers questions using text from the provided context. Be sure to respond in a complete sentence, being                                    comprehensive, including all relevant background information.
    
   Information to read this book.This book has 18 chapters. Each chapter has heading title where each title has number of "TEXT" messages in sanskrit and its english transliteration. Each TEXT contains "SYNONYMS" , "TRANSLATION" ,and "PURPORT".Each SYNONYMS explains meaning of each sanskrit words in a TEXT message. Each TRANSLATION provides english translation of TEXT message. Each PURPORT explains meaning of the TEXT message. 

   Always show full titles against chapters while answering questions related to a particular chapter. 
	  
   If the context is irrelevant to the answer, you may ignore it \n\n


    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    ##model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3) ##
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) 
    

    print("===========End get_conversational_chain")
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    #print("user_question "+user_question) ###	

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    #print("user_question "+user_question) ###	
    #print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Bhagavad Gita Interactive", page_icon = ":surfer:")
    #st.header("Welcome !! Jai Shree Krishna")
    st.image("img/Krish.jpg",width=300)
    #current_datetime = datetime.datetime.now()
    #print(current_datetime)
    user_question = st.text_input("Explore the BG Gen Z way... Ask me !")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.html("<marquee>The project's purpose is to provide an opportunity for Gen Z, the new generation to explore Bhagavad Gita in a trendy way with Generative AI</marquee>")
        st.image("img/Hi.jpg") #AF link
        st.write("---")
	
       
        #st.write("---")
        st.image("img/NPI.png")
    	#st.write("test")  # add this line to display the pro info

        

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; center: 0; width: 60%; background-color: ; padding: 15px; text-align: center; font-size: 10px">
            © <a href="https://books.iskconmumbai.com/s/gallery/iskcon-book-store/bhagavad-gita-as-it-is---regular/x7w/product/-OC8fQCNl6vUmeFyXxpY?imageId=-OC8fQCNl6vUmeFyXxpY" target="_blank"> Own a physical copy</a> | It's magical to read this book. Do not take it as another spiritual book. Read it as any newspaper, whatsapp story or instagram update or any random storybook.You will find answers to your life.  
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
