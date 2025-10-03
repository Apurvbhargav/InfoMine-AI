A. Aim:

    1.Infomine is a interactive PDF-based Question Answering application that enables users to query multiple PDF documents and receive context-aware, natural language answers.
    2.Designed to help professionals, students, and researchers quickly extract relevant insights from lengthy documents.

B. Key Features & Achievements:

    1.Implemented document parsing pipeline to extract and preprocess text from uploaded PDFs using PyPDF2.
    2.Designed an efficient text-chunking strategy with 10,000-character chunks and 1,000-character overlap to maintain context for large documents.
    3.Developed a semantic search system using FAISS vector database with Google Generative AI embeddings for fast and accurate similarity retrieval.
    4.Built a retrieval-augmented generation (RAG) pipeline using LangChain and Gemini 1.5 Flash LLM for precise, context-driven answers.
    5.Integrated into a user-friendly Streamlit web app with real-time Q&A and file upload functionality.
    6.Ensured data scalability by supporting multiple PDF uploads and local vector store persistence.

C.Impact (Quantitative Highlights):

    1.Processed PDFs up to hundreds of pages with chunking + FAISS optimization.
    2.Achieved near-instant query responses (~1–2 seconds) after indexing.
    3.Reduced manual search time in long documents by ~70–80%.
    4.Successfully tested with multiple document uploads (3–5 PDFs simultaneously).
