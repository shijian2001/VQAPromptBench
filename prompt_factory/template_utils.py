#### Util Functions for Template Generator ####

from .template_generator import *
from .vqa_meta_data import *
from typing import *
from tqdm import tqdm

class QuestionTemplateGenerator:
    def __init__(self, enable_balanced_pattern: bool=True):
        self.generator = TemplateGenerator(QUESTION_PATTERNS, enable_balanced_pattern=enable_balanced_pattern)

    def generate(self):
        return self.generator.generate()
    
    @property
    def num_all_potential_prompts(self):
        return self.generator.num_all_potential_prompts

class VQATemplateGenerator:
    def __init__(self, enable_context: bool=False, enable_shuffle: bool=False, enable_balanced_pattern: bool=True):
        self.enable_context = enable_context
        self.enable_shuffle = enable_shuffle
        self.question_generator = QuestionTemplateGenerator(enable_balanced_pattern)
        self.choices_template_generator = TemplateGenerator(CHOICES_PATTERNS, enable_balanced_pattern=enable_balanced_pattern)
        if enable_context:
            self.context_template_generator = TemplateGenerator(CONTEXT_PATTERNS, enable_balanced_pattern=enable_balanced_pattern)

    def generate(self):
        question_template = self.question_generator.generate()
        choices_template = self.choices_template_generator.generate()
        
        templates = [question_template, choices_template]
        
        if self.enable_context:
            context_template = self.context_template_generator.generate()
            templates.insert(1, context_template)  # insert context between question and choices

        if self.enable_shuffle:
            random.shuffle(templates)

        return '\n'.join(templates)
    
    @property
    def num_all_potential_prompts(self):
        total = self.question_generator.num_all_potential_prompts * self.choices_template_generator.num_all_potential_prompts
        if self.enable_context:
            total *= self.context_template_generator.num_all_potential_prompts
        return total

def generate_templates_set(template_generator, num_templates: int):
    """Generate a specified number of prompts with no duplicate elements"""
    assert num_templates <= template_generator().num_all_potential_prompts, (
        "The number of generated templates should be less than or equal to the capacity of the template generator."
    )
    templates_set = set()
    with tqdm(total=num_templates, desc="Generating templates") as pbar:
        while len(templates_set) < num_templates:
            template = template_generator().generate()
            if template not in templates_set:
                templates_set.add(template)
                pbar.update(1)
    return list(templates_set)

def assign_templates(num_data: int, templates_set: List[str]) -> List[str]:
    """Assign a fixed set of prompt templates to the data and ensure that each template is sampled."""
    assert num_data >= len(templates_set), (
        "The number of data should be greater than or equal to the number of templates."
    )
    randomized_templates = random.sample(templates_set, len(templates_set))
    all_templates = randomized_templates + random.choices(templates_set, k=num_data - len(templates_set))
    random.shuffle(all_templates)
    assert set(all_templates) == set(templates_set), "Not all templates have been used." 
    return all_templates