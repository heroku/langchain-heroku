# langchain-heroku

This package contains the LangChain integration with Heroku

## Installation

```bash
pip install -U langchain-heroku
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatHeroku` class exposes chat models from Heroku.

```python
from langchain_heroku import ChatHeroku

llm = ChatHeroku()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`HerokuEmbeddings` class exposes embeddings from Heroku.

```python
from langchain_heroku import HerokuEmbeddings

embeddings = HerokuEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`HerokuLLM` class exposes LLMs from Heroku.

```python
from langchain_heroku import HerokuLLM

llm = HerokuLLM()
llm.invoke("The meaning of life is")
```
