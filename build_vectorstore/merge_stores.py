from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embd = OpenAIEmbeddings(model='text-embedding-3-large', base_url='https://svip.xty.app/v1', api_key='your own key')  # Replace 'your own key' with your actual OpenAI API key

base_path = "./faiss_db_merged"

def merge_vectorstores(paths, target_path, embedding_model):
    merged = None
    for i, path in enumerate(paths):
        store = FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        if merged is None:
            merged = store
        else:
            merged.merge_from(store)
    merged.save_local(target_path)
    print("merge over")

merge_vectorstores(
    [f'./faiss_db_part_{i}' for i in range(20)],
    './faiss_db_merged',
    embd
)