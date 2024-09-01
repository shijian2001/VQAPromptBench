from vqa_meta_data import QUESTION_PATTERNS
from template_generator import TemplateGenerator
import random

def question_template_generator():

    # Define question template generator
    question_template_generator = TemplateGenerator(QUESTION_PATTERNS, enable_balanced_pattern=True)

    # Ggenerate diverse VQA question prompts
    question_template = question_template_generator.generate()
    
    return question_template

if __name__ == "__main__":
    question = "How many cats are there in the image?"
    print(question_template_generator().format(question=question))
    # question_template_generator = TemplateGenerator(QUESTION_PATTERNS, "QUESTION_PATTERNS")
    # question_template_generator.visualize_taxonomy()
    # print("Number of all potential question prompts", question_template_generator.num_all_potential_prompts)