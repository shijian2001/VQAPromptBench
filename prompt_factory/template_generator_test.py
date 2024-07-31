from typing import List
from meta_data import QUESTION_PATTERNS, CONTEXT_PATTERNS, CHOICES_PATTERNS
from base import BaseTemplateGenerator, Pattern
import random

def test_for_template_generator(question: str, candidates: List[str], enable_shuffle: bool=False):

    # Define question and choices template generator
    question_template_generator = BaseTemplateGenerator(QUESTION_PATTERNS)
    choices_template_generator = BaseTemplateGenerator(CHOICES_PATTERNS)

    question_template = question_template_generator.generate()
    choices_template = choices_template_generator.generate()
    templates = [question_template, choices_template]

    # Consider different relative element poisitions
    if enable_shuffle:
        random.shuffle(templates)

    prompt_template = '\n'.join(templates)
    
    return prompt_template.format(
        question=question,
        choices=" ".join(candidates)
    )

if __name__ == "__main__":
    # Generate a VQA prompt template randomly
    question = "How many cats are there in the picture?"
    candidates = ["(A) 1", "(B) 2", "(C) 3", "(D) 4",]
    print(test_for_template_generator(question, candidates))
