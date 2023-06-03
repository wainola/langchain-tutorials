from langchain import OpenAI
from langchain import PromptTemplate
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


    