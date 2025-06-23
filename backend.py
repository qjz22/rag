from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS, InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # 导入matplotlib.image用于读取图像
import os
import gradio as gr
import tempfile
import shutil
import json
import numpy as np
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAIError, APIConnectionError
import requests
import httpx

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((OpenAIError, APIConnectionError, requests.exceptions.ConnectionError, httpx.ConnectError))
)
def with_retry_call(func, *args, **kwargs):
    return func(*args, **kwargs)

os.environ["http_proxy"]="http://localhost:7897"
os.environ["https_proxy"]="http://localhost:7897"  # all proxy depends on your own proxy

os.environ['OPENAI_API_KEY']="you own key" # should be your own key
os.environ['OPENAI_API_BASE']="https://api.siliconflow.cn/v1"
DB_PATH = 'build_vectorstore/faiss_db_merged' # no Chinese path, or it will cause error. should be your own path
CORPUS_PATH = './wiki18_10.jsonl' # should be your own path
embd = OpenAIEmbeddings(model='text-embedding-3-large', base_url='https://svip.xty.app/v1', api_key='your own key') # should be your own key

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
                #docs_split = text_splitter.split_documents(batch)
                partial_store = FAISS.from_documents(batch, embedding=embd)
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
        #docs_split = text_splitter.split_documents(batch)
        partial_store = FAISS.from_documents(batch, embedding=embd)
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

try:
    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings=embd,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"Failed to load FAISS vectorstore: {e}, rebuilding...")
    vectorstore = build_faiss_in_chunks(CORPUS_PATH, DB_PATH, embd, batch_size=10000)

if vectorstore is None:
    raise RuntimeError("Vectorstore could not be initialized.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# question_analysis
question_router = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-7B-Instruct", streaming=True)
route_prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or direct answer. \n 
        Here is the user question: {question} \n
        The vectorstore contains documents from Wikipedia. QA pairs that need grounded facts, like 'who', 'what', 'when', 'where', 'how', etc., should be routed to the vectorstore. \n
        If the question needs knowledge from the vectorstore, answer 'vectorstore'. Otherwise, answer 'direct'. Don't answer other more content.""",
        input_variables=["question"],
    )
question_router = route_prompt | question_router

# grade retrieval
retrieval_grader = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-7B-Instruct", streaming=True)
retrieval_grade_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {document} \n\n 
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document is relevant to the question, answer 'yes'. Otherwise, answer 'no'. Don't answer other more content.""",
    input_variables=["document", "question"],
)
retrieval_grader = retrieval_grade_prompt | retrieval_grader

# generator
generator = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-7B-Instruct", streaming=True)
generate_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    Keep the answer brief and concise. Just give the core of the answer without any explanation. For example, Question: who told the story of the prodigal son \nAnswer: Jesus Christ\n\n
    Question: {question} \nContext: {context} \nAnswer:""",
    input_variables=["question", "context"],
)
generator = generate_prompt | generator | StrOutputParser()

# grade hallucination
hallucination_grader = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-7B-Instruct", streaming=True)
hallucination_grade_prompt = PromptTemplate(
    template="""You are a grader assessing whether an LLM answer is grounded in / supported by a set of retrieved facts. \n
    Here is the user question: {question} \n
    Here are the retrieved facts: \n\n {documents} \n\n 
    Here is the LLM answer: {answer} \n
    If the answer is grounded in / supported by the retrieved facts, answer 'yes'. Otherwise, answer 'no'. Don't answer other more content.""",
    input_variables=["question", "documents", "answer"],
)
hallucination_grader = hallucination_grade_prompt | hallucination_grader

# grade answer
answer_grader = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-7B-Instruct", streaming=True)
answer_grade_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is relevant to a question. \n
    Here is the user question: {question} \n
    Here is the LLM answer: {answer} \n
    Your standard for relevance is whether the answer is possible. Not specific is allowed. Not concise is allowed. Maybe incorrect is allowed. 
    For example, question 'who told the story of the prodigal son' with an answer 'Jesus Christ' is ok, but '1999' is wrong because it's a time and not a person. And question 'when was the second edition of the PDC World Youth Championship' with answer '2012' is ok even though it is not specific. \n
    If the answer is relevant to the question, answer 'yes'. Otherwise, answer 'no'. Don't answer other more content.""",
    input_variables=["question", "answer"],
)
answer_grader = answer_grade_prompt | answer_grader

# rewrite question
question_rewriter = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-7B-Instruct", streaming=True)
question_rewrite_prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval and llm understanding. Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question: \n\n {question} \n Formulate an improved question. Only give the new question, don't include any other content.""",
    input_variables=["question"],
)
question_rewriter = question_rewrite_prompt | question_rewriter | StrOutputParser()

# direct answer
llm = ChatOpenAI(temperature=0, model="Qwen/Qwen2.5-7B-Instruct", streaming=True)
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Keep the answer brief and concise. Just give the core of the answer without any explanation. For example, Question: who told the story of the prodigal son \nAnswer: Jesus Christ\n\n
    Question: {question} \nAnswer:""",
    input_variables=["question"],
)
chatbot = prompt | llm | StrOutputParser()


# state
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    turns: int
    retrieve_turns: int

def initialize_state(question):
    state = GraphState()
    state["question"] = question
    state["documents"] = None
    state["generation"] = None
    state["turns"] = 0
    state["retrieve_turns"] = 0
    return state


# nodes
def retrieve(state):
    global retriever
    print("---RETRIEVE---")
    question = state["question"]
    #documents = retriever.invoke(question)
    documents = with_retry_call(retriever.invoke, question)
    return {"documents": documents, "question": question, "retrieve_turns": state["retrieve_turns"] + 1}

def generate(state):
    global generator
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    #generation = generator.invoke({"context": documents, "question": question})
    generation = with_retry_call(generator.invoke, {"context": format_docs(documents), "question": question})
    return {"documents": documents, "question": question, "generation": generation, "turns": state["turns"] + 1}

def grade_documents(state):
    global retrieval_grader
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        # message = retrieval_grader.invoke(
        #     {"question": question, "document": d.page_content}
        # )
        message = with_retry_call(retrieval_grader.invoke, {"question": question, "document": d.page_content})
        grade = message.content
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    global question_rewriter
    print("---TRANSFORM QUERY---")
    question = state["question"]
    #better_question = question_rewriter.invoke({"question": question})
    better_question = with_retry_call(question_rewriter.invoke, {"question": question})
    return {"question": better_question}

def direct_answer(state):
    global chatbot
    print("---DIRECT ANSWER---")
    #answer = chatbot.invoke({"question": state["question"]})
    answer = with_retry_call(chatbot.invoke, {"question": state["question"]})
    return {"generation": answer}

# edges
def route_question(state):
    global question_router
    print("---ROUTE QUESTION---")
    question = state["question"]
    #source = question_router.invoke({"question": question}).content
    source = with_retry_call(question_router.invoke, {"question": question}).content
    if source == "direct":
        print("---ROUTE QUESTION TO DIRECT ANSWER---")
        return "direct_answer"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"
    else:
        raise ValueError("Invalid source")
    
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION---"
        )
        if state["retrieve_turns"] > 10:
            if state["generation"] is None:
                print("---NO CURRENT ANSWER FOUND---")
                return "direct_answer"
            else:
                print("---CURRENT ANSWER FOUND---")
                return END
        else:
            return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"
    
def check_hallucination_and_answer(state): # TODO: (Optional) change it to a node to pass more information to reflexion
    global hallucination_grader, answer_grader
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # grade = hallucination_grader.invoke(  
    #     {"question": question, "documents": documents, "answer": generation}
    # ).content
    grade = with_retry_call(hallucination_grader.invoke, {"question": question, "documents": format_docs(documents), "answer": generation}).content
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        #grade = answer_grader.invoke({"question": question, "answer": generation}).content
        grade = with_retry_call(answer_grader.invoke, {"question": question, "answer": generation}).content
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return END
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            if state["turns"] <= 1:
                return "generate"
            elif state["turns"] > 6:
                print("---DECISION: MAX TURNS REACHED, END---")
                return END
            else:
                return "transform_query"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        if state["turns"] <= 3:
            return "generate"
        elif state["turns"] > 6:
            print("---DECISION: MAX TURNS REACHED, END---")
            return END
        else:
            return "transform_query"
    
# graph
class Workflow:
    def __init__(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generate
        workflow.add_node("transform_query", transform_query)  # transform_query
        workflow.add_node("direct_answer", direct_answer)  # direct_answer

        workflow.add_conditional_edges(
            START,
            route_question,
            ["direct_answer", "retrieve"]
        )
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            ["transform_query", "generate", "direct_answer", END]
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            check_hallucination_and_answer,
            ["generate", "transform_query", END]
        )
        self.workflow = workflow
        self.graph = self.workflow.compile()

    def show_graph(self):
        try:
            # 使用 Mermaid 生成图表并保存为文件
            mermaid_code = self.graph.get_graph(xray=True).draw_mermaid_png()
            with open("graph.jpg", "wb") as f:
                f.write(mermaid_code)
            # 使用 matplotlib 显示图像
            img = mpimg.imread("graph.jpg")
            plt.imshow(img)
            plt.axis('off')  # 关闭坐标轴
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")


# runner
def runner(question=None):
    if question=='':
        print("ERROR no question")
        return "ERROR no question"
    initial_state = initialize_state(question)
    agent = Workflow()
    # agent.show_graph()
    result = agent.graph.stream(initial_state,config={"recursion_limit": 50},stream_mode="values")
    for event in result:
        print(event)
        last_state = event
    result = last_state['generation']
    return result

def main():
    with gr.Blocks() as demo:
        question = gr.Textbox(label="Question")
        result = gr.Textbox(value="", label="Result")
        btn = gr.Button(value="Submit")
        btn.click(runner,inputs = question, outputs=result)
        clear = gr.ClearButton([question,result])
        demo.launch(share=True)
if __name__ == "__main__":
    main() # UI界面用main()，命令行用runner(question)
    #runner('a political leader during the roman empire was called') # UI界面用main()，命令行用runner(question)
