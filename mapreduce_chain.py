from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.documents import Document

# Define map and reduce templates
map_prompt_template = """Summarize the following chunk:
{chunk}
SUMMARY:"""
reduce_prompt_template = """Combine the following summaries into a final summary:
{summaries}
FINAL SUMMARY:"""
map_prompt = PromptTemplate.from_template(map_prompt_template)
reduce_prompt = PromptTemplate.from_template(reduce_prompt_template)

# Initialize the LLM
openai_api_key = "sk-..."
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)

# Create the map and reduce chains
map_chain = LLMChain(llm=llm, prompt=map_prompt)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Create the MapReduce chain
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="summaries"
    ),
    document_variable_name="chunk",
    return_intermediate_steps=False
)

# Sample large document 
documents = [
    Document(page_content=doc["page_content"], metadata=doc["metadata"])
    for doc in [
        {"page_content": "Apples are red and delicious.", "metadata": {"title": "apple_book"}},
        {"page_content": "Blueberries are blue and sweet.", "metadata": {"title": "blueberry_book"}},
        {"page_content": "Bananas are yellow and nutritious.", "metadata": {"title": "banana_book"}},
    ]
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Execute the chain
summary = map_reduce_chain.run(text_chunks)
print(summary)
