from typing import *
import numpy as np
import random
import re

###################################### base.py ######################################

class MetaElement:
    def __init__(self, name: str, candidates: List[str], comment: str="none"):
        self.name = name
        self.candidates = candidates
        self.comment = comment # comments about the element

    @property
    def random_candidate(self):
        return random.choice(self.candidates)

    @property
    def all_candidates(self):
        return self.candidates

class Pattern:
    """
    - Example pattern: {verb} the{is_following}image, answer the question: {{question}}.
    - Only extract {}, instead of {{}}
    - Designed for template of template
    """
    def __init__(self, pattern: str, meta_elements: Optional[List[MetaElement]]=None):
        self.pattern = pattern
        self.meta_elements = meta_elements or []
        self._check_pattern()
    
    def _check_pattern(self):
        placeholders = re.findall(r'(?<!\{)\{([^{}]*)\}(?!\})', self.pattern)
        self.num_placeholders = len(placeholders)
        if self.meta_elements:
            meta_element_names = [element.name for element in self.meta_elements]
            if set(placeholders) != set(meta_element_names):
                raise ValueError("Pattern placeholders do not match meta_elements names")
    
    def get_num_placeholders(self):
        return self.num_placeholders
    
    def fit_pattern(self):
        if not self.meta_elements:
            return self.pattern.format()
        element_dict = {element.name: element.random_candidate for element in self.meta_elements}
        fited = self.pattern.format(**element_dict).strip()
        return fited[0].upper() + fited[1:]

class BaseTemplateGenerator:
    def __init__(self, patterns: List[Pattern], enable_balanced_pattern: bool=True):
        self.patterns = patterns
        self.enable_balanced_pattern = enable_balanced_pattern

    def _get_balanced_pattern_prob(self):
        """The larger the number of placeholders, the greater the probability that this pattern will be sampled."""
        placeholders_lens = [pattern.get_num_placeholders() for pattern in self.patterns]
        weights = (np.array(placeholders_lens) + 1) / (np.sum(np.array(placeholders_lens) + 1))
        return weights
    
    def add_pattern(self, pattern: Pattern):
        self.patterns.append(pattern)

    def generate(self):
        if self.enable_balanced_pattern:
            pattern = random.choices(self.patterns, weights=self._get_balanced_pattern_prob(), k=1)[0]
        else:
            pattern = random.choice(self.patterns)
        return pattern.fit_pattern()
    
###################################### meta_data.py ######################################

# Global meta elements
is_line_breaking = MetaElement("is_line_breaking", ["\n", " "])
is_please = MetaElement("is_please", [" please ", " "])
is_following = MetaElement("is_following", [" following ", " "])

# Basic question patterns
QUESTION_PATTERNS = [
    Pattern(
        "{{question}}"
    ),
    Pattern(
        "Question:{is_line_breaking}{{question}}",
        [is_line_breaking]
    ),
    Pattern(
        "{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}",
        [   
            MetaElement("answer", ["answer", "determine", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
            is_please,
            is_following,
            is_line_breaking
        ]
    ),
    Pattern(
        "{verb}{what_you_see}{article} {image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
        [
            MetaElement("verb", ["Given", "Analyze", "Referring", "Considering"]), 
            MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
            MetaElement("article", ["the", "this"]),
            MetaElement("image", ["image", "picture"]),
            MetaElement("answer", ["answer", "determine", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
            is_line_breaking,
            is_please,
            is_following
        ]
    ),
    Pattern(
        "{prep}{what_you_see}{article} {image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
        [
            MetaElement("prep", ["Based on", "From", "According to"]), 
            MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
            MetaElement("article", ["the", "this"]),
            MetaElement("image", ["image", "picture"]),
            MetaElement("answer", ["answer", "determine", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
            is_line_breaking,
            is_please,
            is_following
        ]
    )
]

# Basic context patterns
CONTEXT_PATTERNS = [
    Pattern(
        "{{context}}"
    ),
    Pattern(
        "{context}:{is_line_breaking}{{context}}",
        [
            MetaElement("context", ["context", "background", "background information", "hint"]),
            is_line_breaking
        ]
    ),
    Pattern(
        "{verb} the{is_following}{context}:{is_line_breaking}{{context}}",
        [   
            MetaElement("verb", ["Consider", "Given"]),
            MetaElement("context", ["context", "background", "background information", "hint"]),
            is_following,
            is_line_breaking
        ]
    ),
    Pattern(
        "{adv} is the {context}:{is_line_breaking}{{context}}",
        [   
            MetaElement("adv", ["Below", "Here"]),
            MetaElement("context", ["context", "background", "background information", "hint"]),
            is_line_breaking
        ]
    ),
]

# Basic chpices patterns
CHOICES_PATTERNS = [
    Pattern(
        "{{choices}}"
    ),
    Pattern(
        "{choices}:{is_line_breaking}{{choices}}",
        [
            MetaElement("choices", ["Choices", "Options", "Selections"]),
            is_line_breaking
        ]
    ),
    Pattern(
        "{adv} are the {choices}:{is_line_breaking}{{choices}}",
        [
            MetaElement("adv", ["Here", "Below"]),
            MetaElement("choices", ["choices", "options", "selections"]),
            is_line_breaking
        ]
    ),
    Pattern(
        "{verb} the{adj}{answer}{from}{to} the question.{choice}{{choices}}",
        [
            MetaElement("verb", ["Make", "Pick", "Indicate", "Select", "Choose"]),
            MetaElement("adj", [" right ", " correct ", " accurate ", " "]),
            MetaElement("answer", ["answer", "choice", "option", "response", "solution"]),
            MetaElement("from", [
                " from the provided options ", " from the options given ",
                " from the provided choices ", " from the choices given ", 
                " from the choices below ", " from the options below ", 
                " from the available choices ", " from the available options ", " "
            ]),
            MetaElement("to", ["to correctly answer", "to answer", "to address", "to respond to"]),
            MetaElement("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
        ]
    )
]

###################################### test ############################################

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
