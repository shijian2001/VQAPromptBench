from functools import reduce
from typing import *
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
    
    @property
    def num_candidates(self):
        return len(self.candidates)

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

    def _is_duplicate(self, placeholders: List[str]) -> bool:
        return len(placeholders) != len(set(placeholders))
    
    def _check_pattern(self):
        self.placeholders = re.findall(r'(?<!\{)\{([^{}]*)\}(?!\})', self.pattern)
        if self._is_duplicate(self.placeholders):
            raise ValueError("Duplicate placeholders are not allowed")
        if self.meta_elements:
            meta_element_names = [element.name for element in self.meta_elements]
            if (set(self.placeholders) != set(meta_element_names)) or (len(self.placeholders) != len(meta_element_names)):
                raise ValueError("Pattern placeholders do not match meta_elements names")
    
    @property
    def num_placeholders(self):
        return len(self.placeholders)

    @property
    def num_potential_prompts(self):
        return reduce(lambda x, y: x * y.num_candidates, self.meta_elements or [], 1)
    
    def fit_pattern(self):
        if not self.meta_elements:
            return self.pattern.format()
        element_dict = {element.name: element.random_candidate for element in self.meta_elements}
        # Ensured that the first letter of the sentence is capitalized
        # Ensured that the generated senetence is striped
        fited = self.pattern.format(**element_dict).strip()
        return fited[0].upper() + fited[1:]

class Node:
    def __init__(self, name: str, pattern: Optional['Pattern'] = None):
        self.name = name
        self.children: List[Node] = []
        self.pattern = pattern # Only leaf nodes have pattern
        self.weight = 1

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def is_leaf(self) -> bool:
        return self.pattern is not None

    def balance_weights(self) -> float:
        if self.is_leaf():
            # Weights are based on the number of placeholders of the pattern
            # Add-one smoothing
            self.weight = self.pattern.num_placeholders + 1
        else:
            self.weight = sum(child.balance_weights() for child in self.children)
        return self.weight

    def traverse(self) -> 'Node':
        if self.is_leaf():
            return self
        next_node = random.choices(self.children, weights=[child.weight for child in self.children], k=1)[0]
        return next_node.traverse()
    
class TemplateGenerator:
    def __init__(self, data: Union[dict, list], name: str = '', enable_balanced_pattern: bool=True):
        self.root = self._build_taxonomy(data, name)
        if enable_balanced_pattern:
            self.root.balance_weights()

    @property
    def num_all_potential_prompts(self) -> int:
        return self._get_total_prompts(self.root)

    def _get_total_prompts(self, node: Node) -> int:
        if node.is_leaf():
            return node.pattern.num_potential_prompts
        else:
            return sum(self._get_total_prompts(child) for child in node.children)

    def _build_taxonomy(self, data: Union[dict, list], name: str = '') -> Node:
        if isinstance(data, dict):
            node = Node(name=name)
            for key, value in data.items():
                child_node = self._build_taxonomy(value, name=key)
                node.add_child(child_node)
            return node
        elif isinstance(data, list):
            parent_node = Node(name=name)
            for i, pattern in enumerate(data):
                pattern_str = f"pattern_{i+1}"
                pattern_node = Node(name=pattern_str, pattern=pattern)
                parent_node.add_child(pattern_node)
            return parent_node
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _find_node_by_path(self, path: str, node: Optional[Node] = None) -> Optional[Node]:
        """
        - Example Path: Taxonomy/Declarative/Simple
        """
        if node is None:
            node = self.root
        parts = path.split('/')
        if parts[0] != node.name:
            return None
        if len(parts) == 1:
            return node
        for child in node.children:
            result = self._find_node_by_path('/'.join(parts[1:]), child)
            if result:
                return result
        return None

    def generate(self, path: Optional[str] = None):
        start_node = self.root
        if path:
            start_node = self._find_node_by_path(path)
            if not start_node:
                raise ValueError(f"Node with path '{path}' not found.")
        leaf = start_node.traverse()
        if leaf.pattern:
            return leaf.pattern.fit_pattern()
        else:
            raise ValueError("Traversal did not result in a valid pattern.")
    
    def visualize_taxonomy(self, node: Optional[Node] = None, level: int = 0):
        if node is None:
            node = self.root
        indent = " " * (level * 4)
        if node.is_leaf():
            print(f"{indent}- {node.name} (weight: {node.weight}): {node.pattern.pattern}")
        else:
            print(f"{indent}+ {node.name} (weight: {node.weight})")
            for child in node.children:
                self.visualize_taxonomy(child, level + 1)