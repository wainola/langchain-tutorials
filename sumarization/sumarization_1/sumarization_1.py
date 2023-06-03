from dotenv import load_dotenv
import os
from utils import basic_sumarization_prompt, couple_of_paragraphs_summarization

load_dotenv()

open_ai_key = os.getenv("OPENAI_API_KEY")

# Basic level: sumarize couple of sentences
# basic_sumarization_prompt(open_ai_key)

# Sumarizing paragraphs
couple_of_paragraphs_summarization(open_ai_key)

