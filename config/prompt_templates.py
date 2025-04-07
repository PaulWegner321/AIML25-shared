"""
Prompt templates for Watsonx LLM.
"""

# Translation prompt template
TRANSLATION_PROMPT = """
Translate the following ASL tokens to English:

ASL Tokens: {tokens}

Translation:
"""

# Judgment prompt template
JUDGMENT_PROMPT = """
Evaluate the following ASL translation:

Original ASL tokens: {tokens}
Translation: {translation}

Provide feedback on the accuracy and quality of the translation, and assign a score from 0.0 to 1.0.
"""

# RAG prompt template
RAG_PROMPT = """
Answer the following question using the provided context:

Context:
{context}

Question: {question}

Answer:
""" 