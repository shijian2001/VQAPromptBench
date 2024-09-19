#### Util Functions for Template Generator ####

from .template_generator import *
from .vqa_meta_data import *
from typing import *

def get_random_question_template(enable_balanced_pattern: bool=True):
    """Generate a random prompt template with diverse attributes."""
    return TemplateGenerator(QUESTION_PATTERNS, enable_balanced_pattern=enable_balanced_pattern).generate()

def get_random_vqa_template(enable_context: bool=False, enable_shuffle: bool=False, enable_balanced_pattern: bool=True):
    """Generate a random vqa template with diverse attributes."""

    choices_template_generator = TemplateGenerator(CHOICES_PATTERNS, enable_balanced_pattern=enable_balanced_pattern)
    question_template = get_random_question_template()
    choices_template = choices_template_generator.generate()
    templates = [question_template, choices_template]

    if enable_context:
        context_template_generator = TemplateGenerator(CONTEXT_PATTERNS, enable_balanced_pattern=enable_balanced_pattern)
        context_template = context_template_generator.generate()
        templates = [question_template, context_template, choices_template]

    # Consider different relative element poisitions
    if enable_shuffle:
        random.shuffle(templates)

    vqa_prompt_template = '\n'.join(templates)
    return vqa_prompt_template

def generate_templates_set(template_generator: Callable, num_templates: int):
    """Generate a specified number of prompts with no duplicate elements"""
    templates_set = set()
    while len(templates_set) < num_templates:
        template = template_generator()
        templates_set.add(template)
    return list(templates_set)

def assign_templates(num_data: int, template_pool: List[str]) -> List[str]:
    """Assign a fixed set of prompt templates to the data and ensure that each template is sampled."""
    assert num_data >= len(template_pool), (
        "The number of items should be greater than or equal to the number of templates."
    )
    randomized_templates = random.sample(template_pool, len(template_pool))
    all_templates = randomized_templates + random.choices(template_pool, k=num_data - len(template_pool))
    random.shuffle(all_templates)
    assert set(all_templates) == set(template_pool), "Not all templates have been used." 
    return all_templates