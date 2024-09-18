from .template_generator import MetaElement, Pattern

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