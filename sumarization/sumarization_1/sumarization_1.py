from dotenv import load_dotenv
import os
from utils import basic_sumarization_prompt, couple_of_paragraphs_summarization, map_reduce_summarization, vector_summarization, vector_summarization2

load_dotenv()

open_ai_key = os.getenv("OPENAI_API_KEY")

# Basic level: sumarize couple of sentences
# basic_sumarization_prompt(open_ai_key)

# Sumarizing paragraphs
# couple_of_paragraphs_summarization(open_ai_key)

# Using map reduce technique to summarize large text and avoid issues with token limits
# map_reduce_summarization(open_ai_key)

vector_summarization(open_ai_key)

vector_summarization2(open_ai_key)
