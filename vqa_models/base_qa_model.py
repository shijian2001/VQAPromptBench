import random
from typing import Callable
import diskcache
import numpy as np
import sentence_transformers
import torch
import json

SYSTEM_PROMPT = """You are doing a multiple-choice question. 
You will be given an image, question, context, and corresponding options. 
Please give the best option first, and then add your explanation afterward. 
Please strictly follow the json output format with the two keys: "choice", "explanation"

## example
{
    "choice": "The second image is brighter than the first.",
    "explanation": "The first image shows the underwater world, which has a grayish tone. The second image depicts the forest under the sun, which has a bright luster, so the second image is brighter."
}

{
    "choice": "(B) The first image is brighter than the first.",
    "explanation": "The first image depicts a room filled with sunlight, whereas the second image shows that this is a night scene, so the first image is brighter."
}

{
    "choice": "(C)",
    "explanation": "There are three story fragments in the table in the image. The stories numbered 1 and 2 are both written by British novelists, and only the third fragment is written by an American novelist. Therefore, the correct answer is (C) the third fragment."
}

## Given the following information:"""

def make_options(choices, format='letter'):
	assert format in ['numeric', 'letter']
	if format == 'numeric':
		prefix1 = [str(i + 1) for i in range(len(choices))]
	else:
		prefix1 = [chr(ord("a") + i).upper() for i in range(len(choices))]
	prefix2 = [f"({p})" for p in prefix1]
	return prefix1, prefix2, [f'{p} {c}' for p, c in zip(prefix2, choices)]


def check_contain(answer, options):
	contains = [option.lower().strip() in answer.lower().strip() for option in options]
	if sum(contains) == 1:
		return contains.index(True)
	else:
		return -1

def extraction(answer):
	import re

	question_pattern = re.compile(r"Question:\s*(.*?)(?:\n|$)", re.IGNORECASE)
	question_match = question_pattern.search(answer)
	question = question_match.group(1).strip() if question_match else "none"
	
	choices_pattern = re.compile(r"Choices:\s*(.*?)(?:\n|$)", re.IGNORECASE)
	choices_match = choices_pattern.search(answer)
	choices = choices_match.group(1).strip() if choices_match else "none"
	
	keywords = ["Question", "Context", "Choices"]
	pattern = re.compile(rf"({'|'.join(keywords)}):\s.*?(?:\n|$)", re.IGNORECASE)
	cleaned_answer = pattern.sub("", answer)
	
	return cleaned_answer.strip(), question.strip(), choices.strip()

class Model:
	pass

class QAModelInstance:
	def qa(self, data, prompt):
		"(Abstract method) abstract QA method"
	
	def batch_qa(self, data, prompt):
		"(Abstract method) abstract batch QA method"


class QAModel(Model):
	def __init__(
			self,
			model_name: str,
			prompt_func: Callable,
			choice_format='letter',
			enable_choice_search: bool = False,
			cache_path: str = None,
			enable_interpretation: bool = False,
			use_lora: bool = False
	):
		self.model = None
		self.model_name = model_name
		self.prompt_func = prompt_func
		self.format = choice_format
		self.cache_path = cache_path
		self.use_lora = use_lora

		if self.cache_path is not None:
			print(f"[IMPORTANT] model cache is enabled, cache path: {cache_path}")
		else:
			print("[IMPORTANT] model cache is disabled")

		self.enable_interpretation: bool = enable_interpretation
		if self.enable_interpretation:
			print(f"[Interpretable] Start Interpretable Mode")

		self.enable_choice_search = enable_choice_search
		# if enable_choice_search:
			# use SBERT to find the closest choice
		self.sentence_transformer = sentence_transformers.SentenceTransformer("all-mpnet-base-v2", device='cpu')

	@torch.no_grad()
	def choice_search(self, free_form_answer, choices, threshold=0.8):
		query_embedding = self.sentence_transformer.encode([free_form_answer])
		choices_embedding = self.sentence_transformer.encode(choices)
		# similarities = np.dot(choices_embedding, query_embedding.T)
		# top_choice_index = np.argmax(similarities) if similarities[np.argmax(similarities)] >= threshold else -1
		# return choices[top_choice_index] if top_choice_index != -1 else ""
		top_choice_index = np.argmax(np.dot(choices_embedding, query_embedding.T))
		return choices[top_choice_index]

	def _data_to_str(self, data):
		""" abstract method """

	@torch.no_grad()
	def _qa(self, data, prompt, batched=False):
		if self.cache_path is None:
			if batched:
				return self.model.batch_qa(data, prompt)
			else:
				return self.model.qa(data, prompt)
		else:
			with diskcache.Cache(self.cache_path, size_limit=10 * (2 ** 30)) as cache:
				key = json.dumps([self._data_to_str(data), prompt])
				response = cache.get(key, None)
				if response is None:
					if batched:
						response = self.model.batch_qa(data, prompt)
					else:
						response = self.model.qa(data, prompt)
					cache.set(key, response)
				return response

	# Add optional para: prompt_func
	# TODO: Need update for direct QA
	@torch.no_grad()
	def qa(self, data, question, prompt_func=None):
		# prompt = prompt_func(question) if prompt_func else self.prompt_func(question)
		prompt = question
		return self._qa(data, prompt)
	
	@torch.no_grad()
	# def _get_explanation(self, data, qa_prompt: str, multiple_choice_answer: str):
	# 	""" abstract method """
	def _get_explanation(self, data, prompt, free_form_answer: str, max_retry=3):
		for _ in range(max_retry):
			try:
				free_form_answer = json.loads(free_form_answer)
				choice, explanation = free_form_answer["choice"], free_form_answer["explanation"]
				return choice, explanation
			except Exception:
				free_form_answer = self._qa(data, prompt).strip()
		return None, None

	def _limit_answer(self, free_form_answer, choices, prefix1, prefix2, options):
		# Limit the answer to the choices
		if free_form_answer in choices:
			multiple_choice_answer = free_form_answer
		elif free_form_answer in options:
			multiple_choice_answer = choices[options.index(free_form_answer)]
		elif free_form_answer in prefix1:
			multiple_choice_answer = choices[prefix1.index(free_form_answer)]
		elif free_form_answer in prefix2:
			multiple_choice_answer = choices[prefix2.index(free_form_answer)]
		elif self.enable_choice_search:
			multiple_choice_answer = self.choice_search(free_form_answer, choices)
		else:
			multiple_choice_answer = ""
			for to_check in [choices, options, prefix1, prefix2]:
				idx = check_contain(free_form_answer, to_check)
				if idx != -1:
					multiple_choice_answer = choices[idx]
					break
		return multiple_choice_answer

	# Add optional para: prompt_func
	@torch.no_grad()
	def multiple_choice_qa(self, data, question, context, choices, prompt_func=None, answer=None):
		# Get VQA model's answer
		prefix1, prefix2, options = make_options(choices, self.format)
		prompt = prompt_func(question, context, options) if prompt_func else self.prompt_func(question, context, options)
		if self.enable_interpretation:
			prompt = SYSTEM_PROMPT + "\n" + prompt
		free_form_answer = self._qa(data, prompt)
		free_form_answer = free_form_answer.strip()

		if self.enable_interpretation:
			choice, explanation = self._get_explanation(data, prompt, free_form_answer)
			if choice is not None:
				free_form_answer = choice
			if explanation is None:
				explanation = "nan"
				print("[Error] Fail to get explanation")

		# Limit the answer to the choices
		multiple_choice_answer = self._limit_answer(free_form_answer, choices, prefix1, prefix2, options)

		# result = {
		# 	"free_form_answer"      : f"{free_form_answer}\nExplanation: {explanation}" if self.enable_interpretation else free_form_answer,
		# 	"multiple_choice_answer": multiple_choice_answer,
		# 	"choices"               : choices.copy(),
		# }
		result = {
			"free_form_answer"      : free_form_answer,
			"explanation:"          : explanation if self.enable_interpretation else "nan",
			"multiple_choice_answer": multiple_choice_answer,
			"choices"               : choices.copy(),
		}

		if answer is not None:
			result["accuracy"] = int(answer == multiple_choice_answer)
		return result
	
	@torch.no_grad()
	def batch_multiple_choice_qa(self, images, questions, contexts, choices, prompt_func=None, answers=None):
		# Get VQA model's answer
		prefixs1, prefixs2, options = map(list, zip(*[make_options(choice, self.format) for choice in choices]))
		prompts = [
			prompt_func(question, context, option) if prompt_func else self.prompt_func(question, context, option) 
			for question, context, option in zip(questions, contexts, options)
		]
		free_form_answers = self._qa(images, prompts, batched=True)
		free_form_answers = [free_form_answer.strip() for free_form_answer in free_form_answers]

		# disable explaination

		# Limit the answer to the choices
		multiple_choice_answers = [
			self._limit_answer(free_form_answer, choice, prefix1, prefix2, option)
			for free_form_answer, choice, prefix1, prefix2, option
			in zip(free_form_answers, choices, prefixs1, prefixs2, options)
		]

		results = [
			{
				"free_form_answer"      : free_form_answer,
				"multiple_choice_answer": multiple_choice_answer,
				"choices"               : choice.copy(),
			}
			for free_form_answer, multiple_choice_answer, choice
			in zip(free_form_answers, multiple_choice_answers, choices)
		]

		if answers is not None:
			for result, answer in zip(results, answers):
				result["ground_truth_answer"] = answer
				result["accuracy"] = int(answer == result["multiple_choice_answer"])

		return results

	# Add optional para: prompt_func
	@torch.no_grad()
	def multiple_choice_qa_random_ordering(self, data, question, context, choices, prompt_func=None, answer=None, n_trials=3):
		results = {}
		accuracy = 0
		for i in range(n_trials):
			choices_i = choices.copy()
			random.shuffle(choices_i)
			results[i] = self.multiple_choice_qa(data, question, context, choices_i, prompt_func, answer)
			accuracy += results[i]["accuracy"]
		results["accuracy"] = accuracy / n_trials
		return results

	@torch.no_grad()
	def batch_multiple_choice_qa_random_ordering(self, images, questions, contexts, choices, prompt_func=None, answers=None, n_trials=3):
		assert len(images) == len(questions) == len(contexts) == len(choices), "All lengths must the the same"
		batch_size = len(questions)
		batch_results = [{} for _ in range(batch_size)]
		for i in range(n_trials):
			choices_i = choices.copy()
			for choice_i in choices_i:
				random.shuffle(choice_i)
			batch_results_i = self.batch_multiple_choice_qa(images, questions, contexts, choices_i, prompt_func, answers)
			for single_results, single_results_i in zip(batch_results, batch_results_i):
				single_results[i] = single_results_i
	
		for single_results in batch_results:
			single_results["accuracy"] = np.mean([result["accuracy"] for result in single_results.values()])
		
		return batch_results
	
	@torch.no_grad()
	def batch_qa_extraction(self, images, questions, contexts, choices, prompt_func=None, answers=None):
		# Get VQA model's answer
		prefixs1, prefixs2, options = map(list, zip(*[make_options(choice, self.format) for choice in choices]))
		prompts = [
			prompt_func(question, context, option) if prompt_func else self.prompt_func(question, context, option) 
			for question, context, option in zip(questions, contexts, options)
		]
		free_form_answers = self._qa(images, prompts, batched=True)
		free_form_answers = [free_form_answer.strip() for free_form_answer in free_form_answers]

		# Extract answers
		# Extract question and choices for reasoning study
		extracted_answers, extracted_questions, extracted_options = map(list, zip(*[extraction(free_form_answer) for free_form_answer in free_form_answers]))

		# Limit the answer to the choices
		multiple_choice_answers = [
			self._limit_answer(free_form_answer, choice, prefix1, prefix2, option)
			for free_form_answer, choice, prefix1, prefix2, option
			in zip(extracted_answers, choices, prefixs1, prefixs2, options)
		]

		results = [
			{
				"free_form_answer"      : free_form_answer,
				"multiple_choice_answer": multiple_choice_answer,
				"choices"               : choice.copy(),
				"question"              : question,
				"extracted_question"    : extracted_question,
				"passed_option"         : option.copy(),
				"extracted_option"      : extracted_option,
			}
			for free_form_answer, multiple_choice_answer, choice, question, extracted_question, option, extracted_option
			in zip(free_form_answers, multiple_choice_answers, choices, questions, extracted_questions, options, extracted_options)
		]

		def matching(origin, extracted):
			origin, extracted = str(origin), str(extracted)
			if origin in extracted or extracted in origin:
				return 1
			# !choice_search need to be updated! don't use now
			elif self.choice_search(extracted, [origin], threshold=0.8) != "":
				return 1
			else:
				return 0

		for result in results:
			result["question_matched"] = matching(result["question"], result["extracted_question"]) if result["extracted_question"] != "none" else 0
			result["option_matched"] = matching(result["passed_option"], result["extracted_option"]) if result["extracted_option"] != "none" else None

		if answers is not None:
			for result, answer in zip(results, answers):
				result["ground_truth_answer"] = answer
				result["accuracy"] = int(answer == result["multiple_choice_answer"])

		return results
		

