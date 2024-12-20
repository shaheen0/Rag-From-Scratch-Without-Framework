# Rag-From-Scratch-Without-Framework

Retrival Augmented Generation (RAG) is used for information retrival on Documents fancy term of very old concept the field is called information retival
it has two parts

- Knowledge Base Creation
- Generation Part

**1. Knowlegde Base Creation :**
-
- We need knowledge Documents from whcih we want to retrive the certain information
- Chunks them into sub documents
- Each sub documents we compute the  Embeddings [Numerical representation/vectors]
- Store thoes EMbedding + Original Chunks in a vectore store [most of case we dont need vector store]

**2. Generation Part :**
-
- When a new User Query Comes,so we compute embedding for same query
- Then we do Retrival on the vector Store [FIgure out which oe is the closet documents to our User Query]
- In Vector store we find those ,then retrive those,and append thoes to the initial user QUery
- GIves thoes to the LLM  [we get the query+ relevent Document/chunkcs],then LLM generate a response

**Note :**
*there are whloe bounch of Frameworks for implemetation of this Pipeline like Langchain,LlamaIndex popular framework*

**what we need:**
- Python
- Your Embedding Model
- LLM  

