# imports
import os
import glob
import shutil
from dotenv import load_dotenv
import gradio as gr

# imports for langchain - updated to current versions
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
MODEL = "gemini-2.5-flash"  # Using a stable model
DB_NAME = "vector_db"

# Load environment variables
load_dotenv(override=True)

# Ensure API key is set
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

def clean_vector_db():
    """Clean up existing vector database to fix schema issues"""
    if os.path.exists(DB_NAME):
        try:
            print(f"Removing existing vector database at {DB_NAME}")
            shutil.rmtree(DB_NAME)
            print("Vector database cleaned successfully")
        except Exception as e:
            print(f"Warning: Could not remove existing vector database: {e}")

def load_documents():
    """Load and process documents from knowledge base"""
    try:
        # Check if knowledge-base directory exists
        if not os.path.exists("knowledge-base"):
            raise FileNotFoundError("knowledge-base directory not found. Please create it and add your documents.")
        
        # Read in documents using LangChain's loaders
        folders = glob.glob("knowledge-base/*")
        
        if not folders:
            print("Warning: No folders found in knowledge-base/")
            return []
        
        # Text loader configuration for encoding issues
        text_loader_kwargs = {'encoding': 'utf-8'}
       
        documents = []
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder, 
                glob="**/*.md", 
                loader_cls=TextLoader, 
                loader_kwargs=text_loader_kwargs
            )
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        
        print(f"Loaded {len(documents)} documents from {len(folders)} folders")
        return documents
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def create_vectorstore(documents):
    """Create or load vector store"""
    try:
        # Use HuggingFace embeddings (free alternative)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Always clean and recreate to avoid schema issues
        clean_vector_db()
        
        # Split documents into chunks
        if not documents:
            raise ValueError("No documents to process")
            
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        print(f"Created {len(chunks)} chunks from documents")
        
        # Create new vectorstore with a unique collection name
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=DB_NAME,
            collection_name="documents_collection"  # Explicit collection name
        )
        
        print("Vector store created successfully")
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

def create_conversation_chain(vectorstore):
    """Create the conversational retrieval chain"""
    try:
        # Create LLM with proper model name
        llm = ChatGoogleGenerativeAI(
            model=MODEL, 
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True  # Required for Gemini
        )
        
        # Set up conversation memory
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True,
            output_key='answer'
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Return top 3 most relevant chunks
        )
        
        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            memory=memory,
            return_source_documents=True
        )
        
        return conversation_chain
        
    except Exception as e:
        print(f"Error creating conversation chain: {e}")
        raise

def visualize_embeddings(vectorstore, max_points=500):
    """Create t-SNE visualization of document embeddings"""
    try:
        # Get embeddings from vectorstore
        collection = vectorstore._collection
        results = collection.get(include=['embeddings', 'metadatas'])
        
        if not results['embeddings']:
            print("No embeddings found in vector store")
            return None
            
        embeddings_array = np.array(results['embeddings'][:max_points])
        metadatas = results['metadatas'][:max_points]
        
        # Check if we have enough data for t-SNE
        if len(embeddings_array) < 2:
            print("Not enough data points for visualization")
            return None
        
        # Perform t-SNE
        perplexity = min(30, len(embeddings_array)-1)
        if perplexity < 1:
            print("Not enough data points for t-SNE visualization")
            return None
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # Create colors for different document types
        doc_types = [meta.get('doc_type', 'unknown') for meta in metadatas]
        unique_types = list(set(doc_types))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        color_map = {doc_type: colors[i % len(colors)] for i, doc_type in enumerate(unique_types)}
        
        # Create plotly figure
        fig = go.Figure()
        
        # Properly handle the boolean masking
        for doc_type in unique_types:
            # Create indices for this document type
            indices = [i for i, dt in enumerate(doc_types) if dt == doc_type]
            
            if indices:  # Only add trace if we have points for this type
                x_vals = [embeddings_2d[i, 0] for i in indices]
                y_vals = [embeddings_2d[i, 1] for i in indices]
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    name=doc_type,
                    marker=dict(color=color_map[doc_type], size=8),
                    text=[f"Type: {doc_type}" for _ in range(len(x_vals))],
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Document Embeddings Visualization (t-SNE)',
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            hovermode='closest',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main initialization
def initialize_system():
    """Initialize the RAG system"""
    try:
        # Load documents
        documents = load_documents()
        if not documents:
            print("Warning: No documents loaded. The chatbot will have limited functionality.")
            return None, None
        
        # Create vector store
        vectorstore = create_vectorstore(documents)
        
        # Create conversation chain
        conversation_chain = create_conversation_chain(vectorstore)
        
        return conversation_chain, vectorstore
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Initialize the system
print("Initializing RAG system...")
conversation_chain, vectorstore = initialize_system()

if conversation_chain is not None:
    print("System initialized successfully!")
else:
    print("System initialization failed!")

def chat(message, history):
    """Chat function for Gradio interface"""
    if conversation_chain is None:
        return "System not initialized. Please check your setup and restart."
    
    try:
        result = conversation_chain.invoke({"question": message})
        
        # Extract answer and optionally show sources
        answer = result["answer"]
        
        # Optionally add source information
        if "source_documents" in result and result["source_documents"]:
            sources = set()
            for doc in result["source_documents"]:
                if "source" in doc.metadata:
                    sources.add(os.path.basename(doc.metadata["source"]))
            
            if sources:
                answer += f"\n\n*Sources: {', '.join(list(sources))}*"
        
        return answer
        
    except Exception as e:
        return f"Error processing your question: {str(e)}"

def create_gradio_interface():
    """Create Gradio interface with additional features"""
    
    def chat_with_visualization(message, history):
        return chat(message, history)
    
    def show_visualization():
        if vectorstore:
            return visualize_embeddings(vectorstore)
        return None
    
    def get_system_status():
        """Get current system status"""
        if conversation_chain is None:
            return "❌ System not initialized"
        else:
            return "✅ System ready"
    
    # Create the chat interface
    with gr.Blocks(title="RAG Chatbot") as full_interface:
        gr.Markdown("# RAG Chatbot with Document Analysis")
        
        # System status
        status = gr.Textbox(
            value=get_system_status(),
            label="System Status",
            interactive=False
        )
        
        with gr.Tabs():
            with gr.Tab("Chat"):
                if conversation_chain is not None:
                    chat_interface = gr.ChatInterface(
                        fn=chat_with_visualization,
                        title="RAG Chatbot",
                        description="Ask questions about your knowledge base documents",
                        examples=[
                            "What topics are covered in the documents?",
                            "Can you summarize the main points?",
                            "What are the key concepts I should know?"
                        ]
                    )
                else:
                    gr.Markdown("## System Not Initialized")
                    gr.Markdown("""
                    The chatbot system could not be initialized. Please check:
                    
                    1. **GOOGLE_API_KEY**: Make sure it's set in your .env file
                    2. **Knowledge Base**: Ensure the 'knowledge-base' folder exists and contains .md files
                    3. **Dependencies**: Make sure all required packages are installed
                    4. **Permissions**: Check that the application has write permissions for the vector database
                    
                    Try restarting the application after fixing these issues.
                    """)
            
            with gr.Tab("Document Visualization"):
                gr.Markdown("## Document Embeddings Visualization")
                if vectorstore is not None:
                    viz_btn = gr.Button("Generate Visualization")
                    viz_plot = gr.Plot()
                    viz_btn.click(show_visualization, outputs=viz_plot)
                else:
                    gr.Markdown("Vector store not available. Please ensure the system is properly initialized.")
            
            with gr.Tab("System Info"):
                gr.Markdown("## System Information")
                gr.Markdown(f"""
                - **Model**: {MODEL}
                - **Vector DB**: {DB_NAME}
                - **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
                - **API Key Status**: {'✅ Set' if GOOGLE_API_KEY else '❌ Not Set'}
                - **Knowledge Base**: {'✅ Found' if os.path.exists('knowledge-base') else '❌ Not Found'}
                """)
    
    return full_interface

# Launch the interface
if __name__ == "__main__":
    # Create and launch the interface regardless of initialization status
    interface = create_gradio_interface()
    
    try:
        interface.launch(
            inbrowser=True,
            share=False,  # Set to True if you want a public link
            server_name="127.0.0.1",
            server_port=7860
        )
    except Exception as e:
        print(f"Error launching interface: {e}")
        print("Try changing the port or checking if port 7860 is already in use.")
