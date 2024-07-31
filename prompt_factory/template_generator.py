from meta_data import QUESTION_PATTERNS, CONTEXT_PATTERNS, CHOICES_PATTERNS
from base import BaseTemplateGenerator, Pattern
import random

def prompt_template_generator(enable_context:bool=True, enable_shuffle: bool=True):

    # Define question/context/choices template generator
    question_template_generator = BaseTemplateGenerator(QUESTION_PATTERNS)
    context_template_generator = BaseTemplateGenerator(CONTEXT_PATTERNS)
    choices_template_generator = BaseTemplateGenerator(CHOICES_PATTERNS)

    # Leverage the element generators above to generate diverse VQA prompts
    question_template = question_template_generator.generate()
    choices_template = choices_template_generator.generate()
    templates = [question_template, choices_template]

    # Consider the context template when data has the component
    if enable_context:
        context_template = context_template_generator.generate()
        templates.insert(1, context_template)

    # Consider different relative element poisitions
    if enable_shuffle:
        random.shuffle(templates)

    prompt_template = '\n'.join(templates)
    
    return prompt_template

if __name__ == "__main__":
    # Generate a VQA prompt template randomly
    print(prompt_template_generator(enable_shuffle=False))