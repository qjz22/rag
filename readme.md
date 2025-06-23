# RAG

## 代码架构

```plian
RAG/
|-- build_vectorstore/
    |-- build_all.py
    |-- build_faiss_single.py
    |-- merge_stores.py
    |-- split_jsonl.py
backend.py # rag 主体

requirements.txt 

wiki18_10.jsonl # 数据集，此处为一个10条的小demo
```

## 如何运行

首先需要将所有需要openai api key 的地方替换成自己的 api key。

构建 vectorstore。

当数据库较大时，使用 build_vectorstore 里的脚本建库效率更高，数据库小时直接运行 backend.py即可。

```bash
cd build_vectorstore
python split_jsonl.py
build_all.py
merge_stores.py
```

建好向量库后即可运行。如果要使用图形化界面，在backend.py中调用 main 函数，否则调用 runner。

```bash
python backend.py
```
