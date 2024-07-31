from base import MetaElement, Pattern

### Please set the "name" of MetaElement the same as the placeholder in the pattern
### Have ensured that the first letter of the sentence is capitalized
### Have ensured the generated senetence is striped

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