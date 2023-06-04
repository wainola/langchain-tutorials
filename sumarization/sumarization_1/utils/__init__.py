from langchain import OpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

def basic_sumarization_prompt(open_ai_api_key: str):
    print("basic summarization prompt")
    llm = OpenAI(temperature=0, openai_api_key=open_ai_api_key)

    prompt = """
    Provide a summary of the following text:

    Text:
    The first feature of Dasein is that it is a “being as an issue for it”. It takes its own being as an issue; for it is ontological being. In other words, it asks questions about its own existence, it is always confronted with the question “what shall I be today, tomorrow or next year?”.

    And these questions are to be answered by oneself, he calls it “mineness”. We have no other way of experiencing ourselves or the world as being in any other mode than our own existence.
    """

    num_tokens = llm.get_num_tokens(prompt)
    print(f"num_tokens: {num_tokens}")

    output = llm(prompt)
    print(f"output: {output}")

    improved_prompt = """
    Provide a summary of the following text.
    Please provide your output as if you were a school teacher.

    Text:
    The first feature of Dasein is that it is a “being as an issue for it”. It takes its own being as an issue; for it is ontological being. In other words, it asks questions about its own existence, it is always confronted with the question “what shall I be today, tomorrow or next year?”.

    And these questions are to be answered by oneself, he calls it “mineness”. We have no other way of experiencing ourselves or the world as being in any other mode than our own existence.
    """

    num_tokens2 = llm.get_num_tokens(improved_prompt)
    print(f"num_tokens: {num_tokens2}")

    output2 = llm(improved_prompt)
    print(f"output: {output2}")


def couple_of_paragraphs_summarization(open_ai_api_key: str):
    llm = OpenAI(temperature=0, openai_api_key=open_ai_api_key)
    paul_graham_essays = [f'{os.getcwd()}/data/PaulGrahamEssaySmall/getideas.txt', f'{os.getcwd()}/data/PaulGrahamEssaySmall/noob.txt']

    essays = []

    #  We create a promtp template with the instructions for the model
    template = """
    Write a sone sentence summary of the following text:

    {essay}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["essay"]
    )

    for file_name in paul_graham_essays:
        with open(file_name, 'r') as f:
            essays.append(f.read())


    for i, essay in enumerate(essays):
        print(f"essay {i + 1}: {essay[:300]}\n")

    for essay in essays:
        summary_prompt = prompt.format(essay=essay)
        num_tokens = llm.get_num_tokens(summary_prompt)

        print(f'This prompt + essay has {num_tokens} tokens')

        summary = llm(summary_prompt)

        print(f"summary: {summary.strip()}")
        print("\n")

def map_reduce_summarization(openai_api_key: str):

    """
    If we have a text with multiple pages and we would like to summarize that, there is a chance that we will run into token limits, although this is not always the case nor it is a problem. 

    Map reduce type chain is a method that allows to generate a summary of smaller chunks that fits into the token limits. And we can get a summary of the summaries
    """

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    paul_graham_essay = f'{os.getcwd()}/data/PaulGrahamEssays/startupideas.txt'

    with open(paul_graham_essay, 'r') as file:
        essay = file.read()
        num_of_tokens = llm.get_num_tokens(essay)

        print(f"essay has {num_of_tokens} tokens")

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
        docs = text_splitter.create_documents([essay])

        num_docs = len(docs)
        num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

        print(f"num_docs: {num_docs}")
        print(f"num_tokens_first_doc: {num_tokens_first_doc}")

        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce')

        output = summary_chain.run(docs)

        print(f"output: {output}")

        # Getting bullet points for the summary
        map_prompt = """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:
        """

        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

        combine_prompt = """
        Write a concise summary of the following text delimited by triple backquotes:
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
        """

        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template, combine_prompt=combine_prompt_template)

        output2 = summary_chain.run(docs)

        print(f"output2: {output2}")

def vector_summarization(openai_api_key: str):
    """
    The above functions we passed the entire document with more than 9k tokens. Vector summarization covers the use case when you have more than 9k tokens and you want to summarize it.
    """

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    loader = PyPDFLoader(f'{os.getcwd()}/data/IntoThinAirBook.pdf')

    pages = loader.load()

    pages = pages[26:277]

    text = ""

    for page in pages:
        text += page.page_content
    
    text = text.replace("\t", " ")

    num_tokens = llm.get_num_tokens(text)

    print(f"num_tokens: {num_tokens}")
    

def vector_summarization2(openai_api_key: str):
    """
    This function is going to test the vector summarization on a smaller text
    """

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    loader = PyPDFLoader(f'{os.getcwd()}/data/historia.pdf')

    pages = loader.load()

    text = ""

    for page in pages:
        text += page.page_content
    
    text = text.replace("\t", " ")

    num_tokens = llm.get_num_tokens(text)
    
    print(f"num_tokens: {num_tokens}")