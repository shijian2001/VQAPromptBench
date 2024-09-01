from template_generator import MetaElement, Pattern

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
                    "The question {given} {below} is what you should {answer} {considering}{what_you_see}the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [

                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("answer", ["answer", "determine the answer", "respond to", "give your answer", "provide your answer"]),
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
                    "The question {which} {adjective}{is_provided}image is{is_as_follows}{is_line_breaking}{{question}}",
                    [
                        MetaElement("which", ["which", "that"]),
                        MetaElement("adjective", ["is based on the", "is related to the", "you should refer to the", "you should consider the"]),
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
            "Subject-Predicate":[
                Pattern(
                    "{is_please}{answer_directly}{based_image}{is_line_breaking}{{question}}",
                    [   
                        MetaElement("answer_directly", ["answer", "reply", "answer directly", "reply directly"]),
                        MetaElement("based_image", [":", " based on the image:", " considering the image:", " referring to the image:", " based on the provided image:", " considering the provided image:", " referring to the provided image:"]),
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
                    "{is_please}{verb}{what_you_see}{article}{is_provided}{image} and {answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["analyze", "refer to", "consider"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
                Pattern(
                    "{is_please}{verb}{what_you_see}{article}{is_provided_1}{image_1} and {answer} the{is_following}question {related_to} the{is_provided_2}{image_2}:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["analyze", "refer to", "consider"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided_1", [" ", " provided "]),
                        MetaElement("image_1", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided_2", [" ", " provided "]),
                        MetaElement("image_2", ["image", "picture"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                )   
            ],
            "Joined-By-Semicolons":[
                Pattern(
                    "{is_please}{verb}{what_you_see}{article}{is_provided}{image}; {answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["analyze", "refer to", "consider"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
                Pattern(
                    "{is_please}{verb}{what_you_see}{article}{is_provided_1}{image_1}; {answer} the{is_following}question {related_to} the{is_provided_2}{image_2}:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["analyze", "refer to", "consider"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided_1", [" ", " provided "]),
                        MetaElement("image_1", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided_2", [" ", " provided "]),
                        MetaElement("image_2", ["image", "picture"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                )  
            ]
        },
        "Complex":{
            "Adverbial-Clauses":[
                Pattern(
                    "{verb}{what_you_see}{article}{is_provided}{image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["Given", "Analyzing", "Referring to", "Considering"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
                Pattern(
                    "{verb}{what_you_see}{article}{is_provided_1}{image_1},{is_please}{answer} the{is_following}question {related_to} the{is_provided_2}{image_2}:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["Given", "Analyzing", "Referring to", "Considering"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided_1", [" ", " provided "]),
                        MetaElement("image_1", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided_2", [" ", " provided "]),
                        MetaElement("image_2", ["image", "picture"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
                Pattern(
                    "{prep}{what_you_see}{article}{is_provided}{image},{is_please}{answer} the{is_following}question:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("prep", ["Based on", "From", "According to"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                ),
                Pattern(
                    "{is_please}{verb}{what_you_see}{article}{is_provided_1}{image_1} and {answer} the{is_following}question {related_to} the{is_provided_2}{image_2}:{is_line_breaking}{{question}}", 
                    [
                        MetaElement("verb", ["analyze", "refer to", "consider"]), 
                        MetaElement("what_you_see", [" ", " what you see in ", " what you have seen in "]),
                        MetaElement("article", ["the", "this"]),
                        MetaElement("is_provided_1", [" ", " provided "]),
                        MetaElement("image_1", ["image", "picture"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided_2", [" ", " provided "]),
                        MetaElement("image_2", ["image", "picture"]),
                        is_line_breaking,
                        is_please,
                        is_following
                    ]
                )
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
                    "The{is_following}{context} {related_to} the{is_provided}{image} {verb} {object}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("verb", ["requires", "needs", "calls for"]),
                        MetaElement("object", ["your consideration", "your attention"]),
                        is_following,
                        is_line_breaking
                    ]
                ),
            ],
            "Subject-LinkingVerb-Complement":[
                 # Omit the linking-verb: Context (is)
                Pattern(
                    "{is_relevant}{context}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("is_relevant", [" ", "relevant", "the relevant"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{is_the_relevant} {context} {related_to} the{is_provided}{image}{is}{is_line_breaking}{{context}}",
                    [
                        MetaElement("is_the_relevant", ["the", "the relevant"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        MetaElement("is", [":", " is:", " is as follows:"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{intro} is the {context}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),   
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "{intro} is the {context} {related_to} the{is_provided}{image}:{is_line_breaking}{{context}}",
                    [
                        MetaElement("intro", ["The following", "Here", "Below", "Presented below", "Presented here"]),   
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
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
                    "The {context} is {given} {below} {conjunction} you should {conider} it for answering the question:{is_line_breaking}{{question}}",
                    [
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("conjunction", ["and", "then"]),
                        MetaElement("conider", ["consider"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "The question is {given} {below} {conjunction} you should {answer} to the question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("conjunction", ["and", "then"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                ),
            ],
            "Joined-By-Semicolons":[
                Pattern(
                    "The question is {given} {below}; you should {answer} to the question:{is_line_breaking}{{question}}",
                    [
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        is_line_breaking
                    ]
                ),
                Pattern(
                    "The question is {given} {below}; you should {answer} to the question {related_to} the{is_provided}{image}:{is_line_breaking}{{question}}",
                    [
                        MetaElement("given", ["provided", "given", "presented"]),
                        MetaElement("below", ["below", "here"]),
                        MetaElement("answer", ["answer", "determine the answer to", "respond to", "give your answer to", "provide your answer to"]),
                        MetaElement("related_to", ["about", "related to", "based on", "concerning", "regarding"]),
                        MetaElement("is_provided", [" ", " provided "]),
                        MetaElement("image", ["image", "picture"]),
                        is_line_breaking
                    ]
                ),
            ],
        },
        "Complex":{
            "Adjective-Clauses":[
                Pattern(
                    "The {context} {which} {adjective}{is_provided}image is{is_as_follows}{is_line_breaking}{{context}}",
                    [
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        MetaElement("which", ["which", "that"]),
                        MetaElement("adjective", ["is based on the", "is related to the", "you should refer to the", "you should consider the"]),
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
                        MetaElement("verb", ["Consider", "Given"]),
                        MetaElement("context", ["context", "background", "background information", "context information", "hint"]),
                        is_please,
                        is_following,
                        is_line_breaking
                    ]
                ),
            ],
            "Subject-LinkingVerb-Complement":[

            ],
            "Subject-Verb-IndirectObject-DirectObject":[

            ],
            "Subject-Verb-Object-Complement":[

            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[

            ],
            "Joined-By-Semicolons":[

            ],
            "Joined-By-Conjunctive-Adverbs":[

            ]
        },
        "Complex":{
            "Noun-Clauses":[

            ],
            "Adverbial-Clauses":[

            ],
            "Adjective-Clauses":[

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
            "Subject-Predicate":[
                # Omit the predicate: Choices (are)
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
            ],
            "Subject-Verb-Object":[

            ],
            "Subject-LinkingVerb-Complement":[

            ],
            "Subject-Verb-Indirect Object-Direct Object":[

            ],
            "Subject-Verb-Object-Complement":[

            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[

            ],
            "Joined-By-Semicolons":[

            ],
            "Joined-By-Conjunctive-Adverbs":[

            ]
        },
        "Complex":{
            "Noun-Clauses":[

            ],
            "Adverbial-Clauses":[

            ],
            "Adjective-Clauses":[

            ],
        },
    },
    "Interrogative":{
        "Simple":{
            "Subject-Predicate":[

            ],
            "Subject-Verb-Object":[

            ],
            "Subject-LinkingVerb-Complement":[

            ],
            "Subject-Verb-Indirect Object-Direct Object":[

            ],
            "Subject-Verb-Object-Complement":[

            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[

            ],
            "Joined-By-Semicolons":[

            ],
            "Joined-By-Conjunctive-Adverbs":[

            ]
        },
        "Complex":{
            "Noun-Clauses":[

            ],
            "Adverbial-Clauses":[

            ],
            "Adjective-Clauses":[

            ],
        },
    },
    "Imperative":{
        "Simple":{
            "Subject-Predicate":[

            ],
            "Subject-Verb-Object":[
                # Omit the subject
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
            ],
            "Subject-LinkingVerb-Complement":[

            ],
            "Subject-Verb-Indirect Object-Direct Object":[

            ],
            "Subject-Verb-Object-Complement":[

            ],
        },
        "Compound":{
            "Joined-By-Coordinating-Conjunctions":[

            ],
            "Joined-By-Semicolons":[

            ],
            "Joined-By-Conjunctive-Adverbs":[

            ]
        },
        "Complex":{
            "Noun-Clauses":[

            ],
            "Adverbial-Clauses":[

            ],
            "Adjective-Clauses":[

            ],
        },
    }
}

# Basic context patterns
CONTEXT_PATTERNS = [

    ##2.2 复合句
    ##2.2.1 使用并列词连接

    # Pattern(
    #     "The {subject} is {participle} {conjunction} {clause2}.{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("subject", ["context", "information", "background"]),
    #         MetaElement("participle", ["provided", "given"]),
    #         MetaElement("conjunction", ["and", "then"]),
    #         MetaElement("clause2", ["you should consider it carefully", "it should be considered thoroughly", "you should ensure it is thoroughly understood"]),
    #         is_line_breaking
    #     ]
    # ),

    ##2.2.2 使用分号连接
    # Pattern(
    #     "The {subject} is {participle}; {clause2}.{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("subject", ["context", "information", "background"]),
    #         MetaElement("participle", ["provided", "given"]),
    #         MetaElement("clause2", ["you should consider it carefully", "it should be considered thoroughly", "you should ensure it is thoroughly understood"]),
    #         is_line_breaking
    #     ]
    # ),

    ##2.4 复杂性从句
    ##2.4.1 名词性从句
    # Pattern(
    #         "You should {verb1} what the {subject} {verb2}.{is_line_breaking}{{context}}",
    #         [
    #             MetaElement("verb", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #             MetaElement("subject", ["context", "information", "background"]),
    #             MetaElement("verb2", ["explains", "talks about", "implies"]),
    #             is_line_breaking
    #         ]
    #     ),

    #2.4.3 形容词性从句
    # Pattern(
    #     "you should {verb} the {object} which {adjective_clause}.{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("verb", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #         MetaElement("object", ["context", "information", "background"]),
    #         MetaElement("adjective_clause", ["provides the necessary details", "is most relevant to your answer"]),
    #         is_line_breaking
    #     ]
    # )

    ##3 祈使句
    ##3.1 简单句
    ##3.1.1 主谓
    # Pattern(
    #     "{is_please}{verb}:{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("verb", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #         is_please,
    #         is_line_breaking
    #     ]
    # )

    ##3.1.2 主谓宾
    # Pattern(
    #     "{is_please}{verb} the{is_following}{context}:{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("verb", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #         MetaElement("context", ["context", "background", "information"]),
    #         is_please,
    #         is_line_breaking,
    #         is_following
    #     ]
    # ),

    ##3.2 复合句
    ##3.2.1 使用并列连词连接
    # Pattern(
    #     "{is_please}{verb} the{is_following}{context} and {clause2}.{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("verb", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #         MetaElement("context", ["context", "background", "information"]),
    #         MetaElement("clause2", ["dive deeper into the details", "ensure all aspects are covered"]),
    #         is_following,
    #         is_please,
    #         is_line_breaking
    #     ]
    # ),

    ##3.2.2 使用分号连接

    # Pattern(
    #     "{is_please}{verb} the{is_following}{context}; {clause2}.{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("verb", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #         MetaElement("context", ["context", "background", "information"]),
    #         MetaElement("clause2", ["dive deeper into the details", "ensure all aspects are covered"]),
    #         is_following,
    #         is_please,
    #         is_line_breaking
    #     ]
    # ),

    ##3.2.3 使用分号和过渡词连接
    # Pattern(
    #         "{is_please}{verb} the{is_following}{context}; {conj}, {clause2}.{is_line_breaking}{{context}}",
    #         [
    #             MetaElement("verb", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #             MetaElement("context", ["context", "background", "information"]),
    #             MetaElement("conj", ["then", "and"]),
    #             MetaElement("clause2", ["dive deeper into the details", "ensure all aspects are covered"]),
    #             is_following,
    #             is_please,
    #             is_line_breaking
    #         ]
    #     ),

    ##3.3 复杂句
    ##3.3.1 名词性从句
    # Pattern(
    #         "{is_please}{verb1} what the {subject} {verb2}.{is_line_breaking}{{question}}",
    #         [
    #             MetaElement("verb1", ["reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #             MetaElement("subject", ["context", "background", "information"]),
    #             MetaElement("verb2", ["explains", "talks about", "implies"]),
    #             is_please,
    #             is_line_breaking
    #         ]
    #     ),


    ##3.3.2 副词性从句
    # Pattern(
    #     "{is_please}{verb1} the {object} after you {verb2} it.{is_line_breaking}{{context}}",
    #     [
    #         MetaElement("verb1", ["understand", "go through"]),
    #         MetaElement("object", ["context", "information", "background"]),
    #         MetaElement("verb2", ["have read", "finish reading"]),
    #         is_please,
    #         is_line_breaking
    #     ]
    # ),

    ##形容词性从句
    # Pattern(
    #         "{is_please}{verb} the {object} that {adjective_clause}.{is_line_breaking}{{context}}",
    #         [
    #             MetaElement("verb", ["use", "reflect on", "consider", "think about", "read", "look at", "focus on", "pay attention to"]),
    #             MetaElement("object", ["background", "context", "information"]),
    #             MetaElement("adjective_clause", ["explains the key points", "is essential for your analysis"]),
    #             is_please,
    #             is_line_breaking
    #         ]
    #     ),


]

# Basic chpices patterns

CHOICES_PATTERNS = [
    ##1 无
    # Pattern(
    #     "{{choices}}"
    # ),
    ##2 陈述句
    ##2.1 简单句
    ##2.1.1 主谓
    #
    # Pattern(
    #     "{choices}:{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("choices", ["Choices", "Options", "Selections"]),
    #         is_line_breaking
    #     ]
    # ),

    ##2.1.2 主谓宾
    # Pattern(
    #     "{pointer} {subject} {verb} available answers.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("pointer", ["the", "these"]),
    #         MetaElement("subject", ["options", "choices", "selections"]),
    #         MetaElement("verb", ["provide", "give", "show", "display"]),
    #         is_line_breaking
    #     ]
    # ),

    #2.1.3 主系表

    # Pattern(
    #     "{adv} are the {choices}:{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("adv", ["Here", "Below"]),
    #         MetaElement("choices", ["choices", "options", "selections"]),
    #         is_line_breaking
    #     ]
    # ),

    ##2.1.4 主谓双宾
    ##无

    ##2.1.5 主谓宾补 （不行就删）
    # Pattern(
    #     "{pointer} {subject} {verb} {object} {complement}.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("pointer", ["the", "these"]),
    #         MetaElement("subject", ["options", "choices", "selections"]),
    #         MetaElement("verb", ["make", "render"]),
    #         MetaElement("object", ["the decision", "the selection"]),
    #         MetaElement("complement", ["clear", "easier"]),
    #         is_line_breaking
    #     ]
    # )

    ##2.2 复合句
    ##2.2.1 使用并列词连接
    # Pattern(
    #     "{pointer} {subject} are {participle} {conjunction} you should {verb} {object}.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("pointer", ["The", "These"]),
    #         MetaElement("subject", ["options", "choices", "selections"]),
    #         MetaElement("participle", ["provided", "listed below"]),
    #         MetaElement("conjunction", ["and", "then","therefore", "moreover"]),
    #         MetaElement("verb", ["select", "choose"]),
    #         MetaElement("object", ["the best one", "carefully", "the best fit", "the right answer"]),
    #         is_line_breaking
    #     ]
    # ),

    ##2.2.2 使用分号连接

    # Pattern(
    #     "{pointer} {subject} are {participle}; you should {verb} {object}.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("pointer", ["The", "These"]),
    #         MetaElement("subject", ["options", "choices", "selections"]),
    #         MetaElement("participle", ["provided", "listed below"]),
    #         MetaElement("verb", ["select", "choose"]),
    #         MetaElement("object", ["the best one", "carefully", "the best fit", "the right answer"]),
    #         is_line_breaking
    #     ]
    # ),

    ##2.2.3 使用分号和过渡词连接
    # Pattern(
    #     "{pointer} {subject} are {participle}; {conjunction}, you should {verb} {object}.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("pointer", ["The", "These"]),
    #         MetaElement("subject", ["options", "choices", "selections"]),
    #         MetaElement("participle", ["provided", "listed below"]),
    #         MetaElement("conjunction", ["and", "then", "therefore", "moreover"]),
    #         MetaElement("verb", ["select", "choose"]),
    #         MetaElement("object", ["the best one", "carefully", "the best fit", "the right answer"]),
    #         is_line_breaking
    #     ]
    # ),


    ##2.4 复杂句
    ##2.4.1 名词性从句
    # Pattern(
    #     "you should {verb} that {noun_clause}.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("verb", ["ensure", "make sure"]),
    #         MetaElement("noun_clause", ["you consider all the options", "your choice is the best"]),
    #         is_line_breaking
    #     ]
    # ),

    ##2.4.2 副词性从句
    # Pattern(
    #         "{main_clause} when {accessory_clause}.{is_line_breaking}{{choices}}",
    #         [
    #             MetaElement("main_clause", ["you should select the correct option", "you should make your choice"]),
    #             MetaElement("time_clause", ["you have considered all options", "you are ready"]),
    #             is_line_breaking
    #         ]
    #     ),


    ##2.4.3 形容词性从句
    # Pattern(
    #         "you should {verb} {article} {object} that {adjective_clause}.{is_line_breaking}{{choices}}",
    #         [
    #             MetaElement("verb", ["pick", "decide on", "select"]),
    #             MetaElement("article", ["the", "your"]),
    #             MetaElement("object", ["choice", "option"]),
    #             MetaElement("adjective_clause", ["best answers the question", "addresses the question most effectively"]),
    #             is_line_breaking
    #         ]
    #     ),

    ##3 祈使句
    ## 3.1 简单句
    ## 3.1.1 主谓
    # Pattern(
    #     "{is_please}{verb}.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("verb", ["choose", "select", "decide"]),
    #         is_please,
    #         is_line_breaking
    #     ]
    # ),

    ## 3.1.2 主谓宾
    # Pattern(
    #         "{is_please}{verb} your preferred {answer}:{is_line_breaking}{{choices}}",
    #         [
    #             MetaElement("verb", ["indicate", "identify", "determine", "choose", "select"]),
    #             MetaElement("answer", ["answer", "choice", "option", "response", "solution"]),
    #             is_please,
    #             is_line_breaking
    #         ]
    #     ),
# Pattern(
    #     "{is_please}{verb} the{adj}{answer}{from}{to} the question.{choice}{{choices}}",
    #     [
    #         MetaElement("verb", ["Make", "Pick", "Indicate", "Select", "Choose"]),
    #         MetaElement("adj", [" right ", " correct ", " accurate ", " "]),
    #         MetaElement("answer", ["answer", "choice", "option", "response", "solution"]),
    #         MetaElement("from", [
    #             " from the provided options ", " from the options given ",
    #             " from the provided choices ", " from the choices given ",
    #             " from the choices below ", " from the options below ",
    #             " from the available choices ", " from the available options ", " "
    #         ]),
    #         MetaElement("to", ["to correctly answer", "to answer", "to address", "to respond to"]),
    #         MetaElement("choice", ["\nChoices: ", "\nOptions: ", "\n"]),
    #         is_please
    #     ]
    # ),


    ## 3.1.3 主系表
    ##无

    ## 3.1.4 主谓双宾
    ## 无

    ## 3.1.5 主谓宾补
    ##无

    ##3.2 复合句
    ##3.2.1 使用并列连词连接
    # Pattern(
    #     "{is_please}{main_clause} {conj} {second_clause}.{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("main_clause", [
    #             "consider all the options",
    #             "analyze the available options",
    #             "examine each option"
    #         ]),
    #         MetaElement("conj", ["and", "then"]),
    #         MetaElement("second_clause", [
    #             "make your selection carefully",
    #             "pick the most suitable answer",
    #             "decide wisely",
    #             "choose the best possible outcome"
    #         ]),
    #         is_please,
    #         is_line_breaking,
    #     ]
    # ),

    ##3.2.2 使用分号连接
    # Pattern(
    #         "{is_please}{main_clause}; {second_clause}.{is_line_breaking}{{choices}}",
    #         [
    #             MetaElement("main_clause", [
    #                 "consider all the options",
    #                 "analyze the available options",
    #                 "examine each option"
    #             ]),
    #             MetaElement("second_clause", [
    #                 "make your selection carefully",
    #                 "pick the most suitable answer",
    #                 "decide wisely",
    #                 "choose the best possible outcome"
    #             ]),
    #             is_please,
    #             is_line_breaking,
    #         ]
    #     ),
    ## 3.2.3 使用分号和连词连接

    # Pattern(
    #         "{is_please}{main_clause}; {conj}, {second_clause}.{is_line_breaking}{{choices}}",
    #         [
    #             MetaElement("main_clause", [
    #                 "consider all the options",
    #                 "analyze the available options",
    #                 "examine each option"
    #             ]),
    #             MetaElement("conj", ["and", "then"]),
    #             MetaElement("second_clause", [
    #                 "make your selection carefully",
    #                 "pick the most suitable answer",
    #                 "decide wisely",
    #                 "choose the best possible outcome"
    #             ]),
    #             is_please,
    #             is_line_breaking,
    #         ]
    #     ),

    ##3.3 复杂句
    ##3.3.1 名词性从句
    # Pattern(
    #         "{is_please}{verb} that {noun_clause}.{is_line_breaking}{{choices}}",
    #         [
    #             MetaElement("verb", ["ensure", "make sure"]),
    #             MetaElement("noun_clause", ["you consider all the options", "your choice is the best"]),
    #             is_please,
    #             is_line_breaking
    #         ]
    #     ),
    #
    ##3.3.2 副词性从句
        # Pattern(
        #     "{is_please}{main_clause} when {accessory_clause}.{is_line_breaking}{{choices}}",
        #     [
        #         MetaElement("main_clause", ["select the correct option", "make your choice"]),
        #         MetaElement("accessory_clause", ["you have considered all options", "you are ready"]),
        #         is_please,
        #         is_line_breaking
        #     ]
        # ),

    ##3.3.3 形容词性从句

    # Pattern(
    #         "{is_please}{verb} {article} {object} that {adjective_clause}.{is_line_breaking}{{choices}}",
    #         [
    #             MetaElement("verb", ["pick", "decide on", "select"]),
    #             MetaElement("article", ["the", "your"]),
    #             MetaElement("object", ["choice", "option"]),
    #             MetaElement("adjective_clause", ["best answers the question", "addresses the question most effectively"]),
    #             is_please,
    #             is_line_breaking
    #         ]
    #     ),

    ##4 疑问句
    ##4.1 简单句
    ##4.1.1 主谓
    ##无

    ##4.1.2 主谓宾
    # Pattern(
    #     "{aux_verb} you {verb} the {adj} {object}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("aux_verb", ["Do", "Can", "Could"]),
    #         MetaElement("verb", ["select", "choose", "pick"]),
    #         MetaElement("adj", ["correct", "best", "right", "most suitable"]),
    #         MetaElement("object", ["option", "solution", "choice", "answer"]),
    #         is_line_breaking
    #     ]
    # ),

    ##4.1.3 主系表
    # Pattern(
    #     "{interrogative} {linking_verb} the {adj} {object}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("interrogative", ["which", "what"]),
    #         MetaElement("linking_verb", ["is", "would be", "seems to be"]),
    #         MetaElement("adj", ["correct", "best", "right", "most suitable"]),
    #         MetaElement("object", ["option", "solution", "choice", "answer"]),
    #         is_line_breaking
    #     ]
    # ),

    ##4.1.4 主谓双宾
    # Pattern(
    #     "{aux_verb} {pointer} {subject} {verb} you the {adj} {object}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("aux_verb", ["Do", "Can", "Could"]),
    #         MetaElement("pointer", ["the", "these"]),
    #         MetaElement("subject", ["options", "choices", "selections"]),
    #         MetaElement("verb", ["provide", "give", "offer"]),
    #         MetaElement("adj", ["correct", "best", "right", "most suitable"]),
    #         MetaElement("object", ["option", "solution", "choice", "answer"]),
    #         is_line_breaking
    #     ]
    # ),

    ##4.1.5 主谓宾补
    ##无

    ##4.2 复合句
    ##4.2.1 使用并列连词连接
    # Pattern(
    #     "{modal} you {verb1} {pointer} {object1} and {modal} you {verb2} the {adj} {object2}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("modal", ["Can", "Could"]),
    #         MetaElement("verb1", ["consider", "think about"]),
    #         MetaElement("pointer", ["the", "these"]),
    #         MetaElement("object1", ["options", "choices", "selections"]),
    #         MetaElement("verb2", ["select", "choose"]),
    #         MetaElement("adj", ["correct", "best", "right", "most suitable"]),
    #         MetaElement("object2", ["answer", "option", "choice"]),
    #         is_line_breaking
    #     ]
    # ),

    ##4.2.2 使用分号连接

# Pattern(
    #     "{modal} you {verb1} {pointer} {object1}; {modal} you {verb2} the {adj} {object2}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("modal", ["Can", "Could"]),
    #         MetaElement("verb1", ["consider", "think about"]),
    #         MetaElement("pointer", ["the", "these"]),
    #         MetaElement("object1", ["options", "choices", "selections"]),
    #         MetaElement("verb2", ["select", "choose"]),
    #         MetaElement("adj", ["correct", "best", "right", "most suitable"]),
    #         MetaElement("object2", ["answer", "option", "choice"]),
    #         is_line_breaking
    #     ]
    # ),

    ##4.2.3 使用分号和连接词
# Pattern(
    #     "{modal} you {verb1} {pointer} {object1}; and, {modal} you {verb2} the {adj} {object2}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("modal", ["Can", "Could"]),
    #         MetaElement("verb1", ["consider", "think about"]),
    #         MetaElement("pointer", ["the", "these"]),
    #         MetaElement("object1", ["options", "choices", "selections"]),
    #         MetaElement("verb2", ["select", "choose"]),
    #         MetaElement("adj", ["correct", "best", "right", "most suitable"]),
    #         MetaElement("object2", ["answer", "option", "choice"]),
    #         is_line_breaking
    #     ]
    # ),

    ##4.3 复杂句
    ##4.3.1 名词性从句
    # Pattern(
    #     "{modal} you decide {noun_clause}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("modal", ["Can", "Could"]),
    #         MetaElement("noun_clause", ["which option is best", "what the right choice should be", "which choice aligns with the question"]),
    #         is_line_breaking
    #     ]
    # )

    ##4.3.2 副词性从句
    # Pattern(
    #     "{modal} you make a choice {adverbial_clause}?{is_line_breaking}{{choices}}",
    #     [
    #         MetaElement("modal", ["Can", "Could"]),
    #         MetaElement("adverbial_clause", ["after considering all options", "when you consider all options",
    #                                          "once you evaluate all options"]),
    #         is_line_breaking,
    #     ]
    # )

    ##4.3.3 形容词性从句
    # Pattern(
    #     "{modal} you {verb} the {object} {adjective_clause}?{is_line_breaking}{{choices}}",
    #     [
    #
    #         MetaElement("modal", ["Can", "Could"]),
    #         MetaElement("verb", ["select", "choose"]),
    #         MetaElement("object", ["selection", "option", "choice"]),
    #         MetaElement("adjective_clause", ["that best meets the question", "which seems most suitable", "that aligns with the question"]),
    #         is_line_breaking
    #     ]
    # )



]