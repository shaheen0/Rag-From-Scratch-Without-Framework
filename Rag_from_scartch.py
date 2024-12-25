import openai
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from wikipediaapi import Wikipedia
import textwrap
from dotenv import load_dotenv

class RagFromScartch():
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("Openai_API_KEY")
        self.wiki = Wikipedia('Myapp/0.0','en')
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2", trust_remote_code=True)
        # self.model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

    def load_wikiperdia_data(self,page_name):
        page = self.wiki.page(page_name)
        data = page.text
        return data.split('/n/n')
    
    def get_document_embedding(self,paragraph):
        return self.model.encode(paragraph,normalize_embeddings=True)
    
    def query_embedding(self,querry):
        return self.model.encode(querry,normalize_embeddings=True)
    
    def rank_similar_documents(self,doc_embed,query_model):
        similarity = np.dot(doc_embed,query_model.T)
        # top_1_k = np.argsort(similarity,axis=0)[-1:].tolist()
        # return top_1_k
        top_3_k = np.argsort(similarity,axis=0)[-3:][::-1].tolist()
        return top_3_k
    
    def generate_prompt(self,context,query):
        return f"""
        use the following CONTEXT to answer the QUESTION at the end.
        if you don't know the answer, just say you don't know, don't try to make up an answer.
        CONTEXT: {context}
        QUESTION: {query}
        """
    
    def get_openai_response(self ,prompt):
        response = openai.chat.completions.create(
            model= "gpt-4",
            messages=[{"role":"user","content":prompt}],
            max_tokens=2000  
        ) 
        return response.choices[0].message.content
    
    def process_data(self,page_name,query):
        paragraph = self.load_wikiperdia_data(page_name)
        doc_embed = self.get_document_embedding(paragraph)
        query_model = self.query_embedding(query)

        top_k = self.rank_similar_documents(doc_embed,query_model)
        most_similar_documents = paragraph[top_k[0]]    # get single one
        # most_similar_documents = [paragraph[idx] for idx in top_k]
        # Wrap the most similar documents into readable format
        context = ""
        for p in most_similar_documents:
            wraped_text = textwrap.fill(p, width=200)
            print("-" * 68)
            print(wraped_text)
            print("-" * 68)
            context += wraped_text + "\n\n"

        prompt = self.generate_prompt(context, query)
        if len(prompt) > 8192:
            prompt = prompt  # Added truncation
        response_content = self.get_openai_response(prompt)
        print(response_content)


# Usage
if __name__ == "__main__":
    llm_query = RagFromScartch()
    llm_query.process_data('Large_language_models', "tell me about LLM ")
