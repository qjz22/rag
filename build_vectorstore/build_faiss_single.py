import sys
import os
from langchain_openai import OpenAIEmbeddings
import tempfile
import shutil
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAIError


embd = OpenAIEmbeddings(model='text-embedding-3-large', base_url='https://svip.xty.app/v1', api_key='your own key') # Replace 'your own key' with your actual OpenAI API key
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

os.environ['OPENAI_API_KEY']="your own key"  # Replace 'your own key' with your actual OpenAI API key
os.environ['OPENAI_API_BASE']="https://api.siliconflow.cn/v1"

@retry(
    reraise=True,
    stop=stop_after_attempt(10),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type((OpenAIError, TypeError))
)
def safe_faiss_from_documents(docs, embd):
    return FAISS.from_documents(docs, embedding=embd)

def build_faiss_in_chunks(corpus_path, db_path, embd, batch_size=1000):
    vectorstore = None
    temp_dir = tempfile.mkdtemp()
    batch = []
    total = 0
    part_id = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            content = obj.get("contents", "")
            batch.append(Document(page_content=content))
            if len(batch) >= batch_size:
                print(f"处理第 {part_id+1} 批...")
                partial_store = safe_faiss_from_documents(batch, embd)
                part_path = os.path.join(temp_dir, f"part_{part_id}")
                partial_store.save_local(part_path)

                if vectorstore is None:
                    vectorstore = partial_store
                else:
                    vectorstore.merge_from(partial_store)

                total += len(batch)
                batch = []
                part_id += 1

    # 处理最后一个 batch
    if batch:
        print(f"处理最后一批...")
        partial_store = safe_faiss_from_documents(batch, embd)
        if vectorstore is None:
            vectorstore = partial_store
        else:
            vectorstore.merge_from(partial_store)
        total += len(batch)

    print(f"共处理 {total} 个文档段落，保存至 {db_path}")
    if vectorstore is not None:
        vectorstore.save_local(db_path)
    else:
        print("Warning: vectorstore is None, skipping save_local.")

    # 清理临时目录
    shutil.rmtree(temp_dir)
    return vectorstore

if __name__ == "__main__":
    corpus_path = sys.argv[1]
    db_path = sys.argv[2]
    build_faiss_in_chunks(corpus_path, db_path, embd, 5000)
