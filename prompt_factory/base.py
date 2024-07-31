from typing import *
import numpy as np
import random
import re

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