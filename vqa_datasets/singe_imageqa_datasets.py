from .base_vqa_datasets import SingleVQADatsetInstance, BaseSingleVQADataset
from datasets import load_from_disk

single_image_qa_datasets = {
    "blink" : ("BLINK", "./subset/blink"),
    "mmbench" : ("MMBench", "./subset/mmbench"),
    "seedbench1": ("SeedBench1", "./subset/seedbench1")
}

class SingleImageQADataset(BaseSingleVQADataset):
    def __init__(
            self, 
            dataset_name: str,
            dataset: SingleVQADatsetInstance = None
        ):
        super().__init__(dataset_name)

        if dataset is None:
            print(f"Loading {dataset_name}...")
            class_name, dataset_path = single_image_qa_datasets[dataset_name]
            self.dataset = eval(class_name)(dataset_path)
            print(f"Finish loading {dataset_name}")

class BLINK(SingleVQADatsetInstance):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(["image_1", "question", "choices", "answer"]).rename_column("image_1", "image")

        def _process_data(sample):

            sample["context"] = ""
            
            # change answer from A/B/C to the concrete value
            answer_to_index = {"(A)": 0, "(B)": 1, "(C)": 2, "(D)": 3}
            index = answer_to_index[sample["answer"]]
            sample["answer"] = sample["choices"][index]

            return sample

        standard_dataset = standard_dataset.map(_process_data)

        return standard_dataset

class MMBench(SingleVQADatsetInstance):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(["image", "question", "hint", "answer", "A", "B", "C", "D"]).rename_column("hint", "context")

        def _process_data(sample):
            sample["choices"] = [sample[option] for option in [ "A", "B", "C", "D"] if sample[option] != "nan"]

            answer_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            index = answer_to_index[sample["answer"]]

            sample["answer"] = sample["choices"][index]

            return sample

        standard_dataset = standard_dataset.map(_process_data)
        standard_dataset = standard_dataset.remove_columns(["A", "B", "C", "D"])
        return standard_dataset

class SeedBench1(SingleVQADatsetInstance):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(["image", "question", "answer", "choice_a", "choice_b", "choice_c", "choice_d"]).rename_column("image", "image_list")

        def _process_data(sample):

            sample["context"] = ""
            
            assert len(sample["image_list"]) == 1
            sample["image"] = sample["image_list"][0]
            sample["choices"] = [sample[option] for option in ["choice_a", "choice_b", "choice_c", "choice_d"]]

            answer_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            index = answer_to_index[sample["answer"]]

            sample["answer"] = sample["choices"][index]

            return sample

        standard_dataset = standard_dataset.map(_process_data)
        standard_dataset = standard_dataset.remove_columns(["choice_a", "choice_b", "choice_c", "choice_d", "image_list"])
        return standard_dataset