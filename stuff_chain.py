from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Define the prompt
prompt_template = """Provide a short summary of the given content:
``{context}``
SHORT SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)
# Initialize the LLM
openai_api_key = "sk-..."
llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=openai_api_key)
# Define StuffDocumentsChain
stuff_chain = create_stuff_documents_chain(llm, prompt)
# Example document
documents = [Document(page_content="What is the Capital of India?")]
summary = stuff_chain.invoke({"context": documents})
print(summary)
