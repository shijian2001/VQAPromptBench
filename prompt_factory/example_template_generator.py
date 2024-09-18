from functools import reduce
from typing import *
import random
import re

###################################### template_generator.py ######################################

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
    
###################################### vqa_meta_data.py ######################################

### Please set the "name" of MetaElement the same as the placeholder in the pattern
### Have ensured that the first letter of the sentence is capitalized
### Have ensured the generated senetence is striped

# Global meta elements
is_line_breaking = MetaElement("is_line_breaking", ["\n", " "])
is_please = MetaElement("is_please", [" please ", " "])
is_following = MetaElement("is_following", [" following ", " "])

# Question Patterns
QUESTION_PATTERNS = {
    "Empty":[
        Pattern(
            "{{question}}"
        ),
    ],
    "Declarative":{
        "Simple":{
            "Subject-Verb-Object":[
                Pattern(
                    "The{is_following}question {related_to} the{is_provided}{image} {verb} {object}:{is_line_breaking}{{question}}",
                    [
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("verb", ["asks for", "requires"]),
                        MetaElement("object", ["your response", "an answer", "an response", "your answer"]),
                        is_following,
                        is_line_breaking
                    ]
                ),

            ],
            "Subject-LinkingVerb-Complement":[
                # Omit the linking-verb: Question (is)
                Pattern(
                    "Question:{is_line_breaking}{{question}}",
                    [
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_the}question {related_to} the{is_provided}{image}{is}{is_line_breaking}{{question}}",
                    [
                        MetaElement("is_the", [" ", " the "]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("is", [":", " is:", " is as follows:"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{intro} is the question:{is_line_breaking}{{question}}",
                    [
                        MetaElement("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{intro} is the question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        MetaElement("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[
                Pattern(
                    "The question is {given} {below} {conjunction} you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("conjunction", ["and", "then"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "The question {related_to} the{is_provided}{image} is {given} {below} {conjunction} you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("conjunction", ["and", "then"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Joined-By-Semicolons":[
                Pattern(
                    "The question is {given} {below}; you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "The question {related_to} the{is_provided}{image} is {given} {below}; you should {answer} it:{is_line_breaking}{{question}}",
                    [
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Complex":{
            "Noun-Clauses":[
                Pattern(
                    "The question {given} {below} is what you should {answer}:{is_line_breaking}{{question}}",
                    [

                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "The question {given} {below} is what you should {answer} {considering}{what_you_see}the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [

                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("considering", ["considering", "based on", "referring to"]),
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Adjective-Clauses":[
                Pattern(
                    "The question {which} {adjective}{is_provided}{image} is{is_as_follows}{is_line_breaking}{{question}}",
                    [
                        MetaElement("which", ["which", "that"]),
                        MetaElement("adjective", ["is based on the", "is related to the", "you should refer to the", "you should consider the"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("is_as_follows",[":", " as follows:"]),
                        is_line_breaking
                    ]
                )
            ],
        },
    },
    "Imperative":{
        "Simple":{
            "Subject-Predicate":[
                Pattern(
                    "{is_please}{answer_directly}:{is_line_breaking}{{question}}",
                    [   
                        MetaElement("answer_directly", ["answer", "reply", "answer directly", "reply directly"]),
                        is_please,
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_please}{answer_directly} {considering}{what_you_see}the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [   
                        MetaElement("answer_directly", ["answer", "reply", "answer directly", "reply directly"]),
                        MetaElement("considering", ["considering", "based on", "referring to"]),
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_please,
                        is_line_breaking
                    ]
                )
            ],
            "Subject-Verb-Object":[
                Pattern(
                    "{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}",
                    [   
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_please,
                        is_following,
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_please}{answer} the{is_following}question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [   
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_please,
                        is_following,
                        is_line_breaking
                    ]
                )
            ],
            "Subject-Verb-IndirectObject-DirectObject":[
                Pattern(
                    "{verb} me {the_answer} to the question:{is_line_breaking}{{question}}",
                    [
                        MetaElement("verb", ["Give", "Provide"]),
                        MetaElement("the_answer", ["your answer", "the correct answer", "a response"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{verb} me {the_answer} to the question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        MetaElement("verb", ["Give", "Provide"]),
                        MetaElement("the_answer", ["your answer", "the correct answer", "a response"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                )
            ]
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[
                Pattern(
                    "{is_please}{verb}{what_you_see}the{is_provided}{image} and {answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["analyze", "refer to", "consider"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),   
            ],
            "Joined-By-Semicolons":[
                Pattern(
                    "{is_please}{verb}{what_you_see}the{is_provided}{image}; {answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["analyze", "refer to", "consider"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),  
            ]
        },
        "Complex":{
            "Adverbial-Clauses":[
                Pattern(
                    "{verb}{what_you_see}the{is_provided}{image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["Given", "Analyzing", "Referring to", "Considering"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
                Pattern(
                    "{prep}{what_you_see}the{is_provided}{image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("prep", ["Based on", "From", "According to"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
            ],
            "Adjective-Clauses":[
                Pattern(
                    "{is_please}{answer} the question {which} {adjective}{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("adjective", ["is based on the", "is related to the", "you should refer to the", "you should consider the"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_please,
                        is_line_breaking
                    ]
                )
            ],
        },
    }
}


# Context Patterns
CONTEXT_PATTERNS = {
    "Empty":[
        Pattern(
            "{{context}}"
        ),
    ],
    "Declarative":{
        "Simple":{
            "Subject-Verb-Object":[
                Pattern(
                    "{the_relevant} {context} {related_to} the{is_provided}{image} {verb} {object}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("the_relevant", ["the", "the relevant"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("verb", ["requires", "needs", "calls for"]),
                        MetaElement("object", ["your consideration", "your attention"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Subject-LinkingVerb-Complement":[
                # Omit the linking-verb: Context (is)
                Pattern(
                    "{is_the_relevant} {context}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("is_the_relevant", [" ", "the", "relevant", "the relevant"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_the_relevant} {context} {related_to} the{is_provided}{image}{is}{is_line_breaking}{{context}}",
                    [
                        MetaElement("is_the_relevant", [" ", "the", "relevant", "the relevant"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("is", [":", " is:", " is as follows:"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{intro} is the{relevant}{context}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        MetaElement("relevant", [" ", " relevant "]),   
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{intro} is the{relevant}{context} {related_to} the{is_provided}{image}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        MetaElement("relevant", [" ", " relevant "]),      
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[
                Pattern(
                    "The{relevant}{context} is {given} {below} {conjunction} you should {verb} it:{is_line_breaking}{{context}}",
                    [
                        MetaElement("relevant", [" ", " relevant "]),      
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("conjunction", ["and", "then"]),
                        MetaElement("verb", ["consider", "pay attention to"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "The{relevant}{context} {related_to} the{is_provided}{image} is {given} {below} {conjunction} you should {verb} it:{is_line_breaking}{{context}}",
                    [
                        MetaElement("relevant", [" ", " relevant "]),      
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("conjunction", ["and", "then"]),
                        MetaElement("verb", ["consider", "pay attention to"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Joined-By-Semicolons":[
                Pattern(
                    "The{relevant}{context} is {given} {below}; you should {verb} it:{is_line_breaking}{{context}}",
                    [
                        MetaElement("relevant", [" ", " relevant "]),      
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("verb", ["consider", "pay attention to"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "The{relevant}{context} {related_to} the{is_provided}{image} is {given} {below}; you should {verb} it:{is_line_breaking}{{context}}",
                    [
                        MetaElement("relevant", [" ", " relevant "]),      
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("verb", ["consider", "pay attention to"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Complex":{
            "Adjective-Clauses":[
                Pattern(
                    "The{relevant}{context} {which} {adjective} the{is_provided}image is{is_as_follows}{is_line_breaking}{{context}}",
                    [
                        MetaElement("relevant", [" ", " relevant "]),      
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("adjective", ["is about", "is related to"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("is_as_follows",[":", " as follows:"]),
                        is_line_breaking
                    ]
                )
            ],
        },
    },
    "Imperative":{
        "Simple":{
            "Subject-Verb-Object":[
                # Omit the subject
                Pattern(
                    "{is_please}{verb} the{is_following}{context}:{is_line_breaking}{{context}}",
                    [   
                        MetaElement("verb", ["consider", "pay attention to"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        is_please,
                        is_following,
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_please}{verb} the{is_following}{context} {related_to} the{is_provided}{image}:{is_line_breaking}{{context}}",
                    [   
                        MetaElement("verb", ["consider", "pay attention to"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_please,
                        is_following,
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Complex":{
            "Adverbial-Clauses":[
                Pattern(
                    "Given the{is_following}{context}:{is_line_breaking}{{context}}",
                    [   
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        is_following,
                        is_line_breaking
                    ]
                ),
            ],
            "Adjective-Clauses":[
                Pattern(
                    "{is_please}{verb} the {context} {which} {adjective} the{is_provided}{image}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("verb", ["consider", "pay attention to"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("adjective", ["is about", "is related to"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_please,
                        is_line_breaking
                    ]
                )
            ],
        },
    }
}

# Choices Patterns
CHOICES_PATTERNS = {
    "Empty":[
        Pattern(
            "{{choices}}"
        ),
    ],
    "Declarative":{
        "Simple":{
            "Subject-LinkingVerb-Complement":[
                # Omit the linkingVerb: Choices (are)
                Pattern(
                    "{is_the}{is_available}{choices}{are}{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("are", [":", " are:"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_the}{is_available}{choices} are as follows:{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_the}{is_available}{choices} are {provided}{below}{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("provided", ["provided", "offered", "presented", "listed"]),
                        MetaElement("below", [":", " below:", " here:", " as follows:"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{adv} are the{is_available}{choices}:{is_line_breaking}{{choices}}",
                    [
                        MetaElement("adv", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[
                Pattern(
                    "{is_the}{is_available}{choices} are {provided} {below} {conjunction} you should {verb} {object}:{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("provided", ["provided", "offered", "presented", "listed"]),
                        MetaElement("below", ["below", "here", "as follows"]),
                        MetaElement("conjunction", ["and", "then"]),
                        MetaElement("verb", ["select", "choose", "pick"]),
                        MetaElement("object", ["the best one", "the best fit", "the best answer"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Joined-By-Semicolons":[
                Pattern(
                    "{is_the}{is_available}{choices} are {provided} {below}; you should {verb} {object}:{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("provided", ["provided", "offered", "presented", "listed"]),
                        MetaElement("below", ["below", "here", "as follows"]),
                        MetaElement("verb", ["select", "choose", "pick"]),
                        MetaElement("object", ["the best one", "the best fit", "the best answer"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Complex":{
            "Adjective-Clauses":[
                Pattern(
                    "{is_the}{is_available}{choices} {which} include only one {correct} answer{are}{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("correct", ["correct", "best", "most suitable"]),
                        MetaElement("are", [":", " are:"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_the}{is_available}{choices} {which} include only one {correct} answer are as follows:{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("correct", ["correct", "best", "most suitable"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_the}{is_available}{choices} {which} include only one {correct} answer are {provided}{below}{is_line_breaking}{{choices}}",
                    [
                        MetaElement("is_the", [" ", "the"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("correct", ["correct", "best", "most suitable"]),
                        MetaElement("provided", ["provided", "offered", "presented", "listed"]),
                        MetaElement("below", [":", " below:", " here:", " as follows:"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{adv} are the{is_available}{choices} {which} include only one {correct} answer:{is_line_breaking}{{choices}}",
                    [
                        MetaElement("adv", ["The following", "Here", "Below", "Presented below", "Presented here"]),
                        MetaElement("is_available", [" ", " available ", " potential ", " possible "]),
                        MetaElement("choices", ["choices", "options", "selections"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("correct", ["correct", "best", "most suitable"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
    },
    "Imperative":{
        "Simple":{
            "Subject-Predicate":[
                Pattern(
                    "{is_please}{verb} from:{choice}{{choices}}",
                    [
                        MetaElement("verb", ["pick", "select", "choose"]),
                        MetaElement("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
                        is_please
                    ]
                )
            ],
            "Subject-Verb-Object":[
                # Omit the subject
                Pattern(
                    "{is_please}{verb} the{adj}{answer}{from}{to} the question.{choice}{{choices}}",
                    [
                        MetaElement("verb", ["make", "pick", "indicate", "select", "choose"]),
                        MetaElement("adj", [" right ", " correct ", " accurate ", " "]),
                        MetaElement("answer", ["answer", "choice", "option", "response", "solution"]),
                        MetaElement("from", [
                            " from the provided options ", " from the options given ",
                            " from the provided choices ", " from the choices given ", 
                            " from the choices below ", " from the options below ", 
                            " from the available choices ", " from the available options ", " "
                        ]),
                        MetaElement("to", [
                            "to correctly answer", "to answer", 
                            "to correctly address", "to address", 
                            "to correctly respond to", "to respond to"
                        ]),
                        MetaElement("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
                        is_please
                    ]
                )
            ],
        },
        "Complex":{
            "Adjective-Clauses":[
                Pattern(
                    "{is_please}{verb} the{adj}{answer}{from}{which} include only one {correct} answer {to} the question.{choice}{{choices}}",
                    [
                        MetaElement("verb", ["make", "pick", "indicate", "select", "choose"]),
                        MetaElement("adj", [" right ", " correct ", " accurate ", " "]),
                        MetaElement("answer", ["answer", "choice", "option", "response", "solution"]),
                        MetaElement("from", [
                            " from the provided options ", " from the options given ",
                            " from the provided choices ", " from the choices given ", 
                            " from the choices below ", " from the options below ", 
                            " from the available choices ", " from the available options ", " "
                        ]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("correct", ["correct", "best", "most suitable"]),
                        MetaElement("to", [
                            "to correctly answer", "to answer", 
                            "to correctly address", "to address", 
                            "to correctly respond to", "to respond to"
                        ]),
                        MetaElement("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
                        is_please
                    ]
                )
            ],
        },
    }
}

###################################### test ############################################

# Utils function
def make_options(choices, format='letter'):
    assert format in ['numeric', 'letter']
    if format == 'numeric':
        prefix1 = [str(i + 1) for i in range(len(choices))]
    else:
        prefix1 = [chr(ord("a") + i).upper() for i in range(len(choices))]
    prefix2 = [f"({p})" for p in prefix1]
    return prefix1, prefix2, [f'{p} {c}' for p, c in zip(prefix2, choices)]

def test_for_template_generator(question: str, candidates: List[str], answer_id: List[int], enable_shuffle: bool=False):

    prefix1, prefix2, options = make_options(candidates)
    answer = random.choice([prefix1[answer_id[0]], prefix2[answer_id[0]], options[answer_id[0]]])

    # Define question and choices template generator
    question_template_generator = TemplateGenerator(QUESTION_PATTERNS, enable_balanced_pattern=True)
    choices_template_generator = TemplateGenerator(CHOICES_PATTERNS, enable_balanced_pattern=True)

    question_template = question_template_generator.generate()
    choices_template = choices_template_generator.generate()
    templates = [question_template, choices_template]

    # Consider different relative element poisitions
    if enable_shuffle:
        random.shuffle(templates)

    prompt_template = '\n'.join(templates)
    
    return prompt_template.format(
        question=question,
        choices=" ".join(options)
    ), answer

if __name__ == "__main__":
    # Given question, candidates, answer_id, generate VQA prompt with random templates
    question = "How many cats are there in the image?"
    candidates = ["1", "2", "3", "4"]
    answer_id = [2]
    prompt, answer = test_for_template_generator(question, candidates, answer_id)
    print("###################################### Prompt #######################################")
    print(prompt)
    print("###################################### Answer #######################################")
    print(answer)

    # Example output
    # ###################################### Prompt #######################################
    # Referring what you see in this image, determine the answer to the following question: How many cats are there in the picture?
    # Indicate the correct solution to answer the question.
    # (A) 1 (B) 2 (C) 3 (D) 4
    # ###################################### Answer #######################################
    # (C) 3