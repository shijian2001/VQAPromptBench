import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical

def evaluator(prompt):
    accuracy = get_model_performance(prompt)
    return -accuracy

def get_model_performance(prompt):
    np.random.seed()
    return np.random.random()

prompt_templates = [
    'Template 1',
    'Template 2',
    'Template 3',
    'Template 4',
    'Template 5'
]
space = [Categorical(prompt_templates, name='prompt templates')]
print(space)

result = gp_minimize(
    func=evaluator,
    dimensions=space,
    acq_func='EI',
    n_calls=10,
    random_state=0
)

best_template_index = np.argmin(result.func_vals)
print(best_template_index)
print(result.x)
best_template = result.x[best_template_index]

print(f"The best prompt template is: {best_template}")
print(f"Performance: {-result.fun}")