import pandas as pd
from data import Data_Preprocessing
from scenario import TEST_1, TEST_2
from prompt_composition import Prompt_Composition
from generator import LLM
from evaluation import EvaluationTools
from config import TEST_CATEGORIES, SAMPLE_COUNT_PER_DATASET
from taxonomy import TAXONOMY_DICT
from colorama import Fore
from tqdm import tqdm

class Guard():
    def __init__(self):
        print(Fore.GREEN + f"Guard processing starts")
    def taxonomy(self, category):
        print(Fore.RED + f"category configured : {category}")
        print(Fore.RESET + f"definition : {TAXONOMY_DICT[category]['definition']}")
        return TAXONOMY_DICT[category]

    def scenario(self, test_case):
        scenario2case_map = {"TEST_1" : TEST_1, "TEST_2" : TEST_2}
        print(f"Running TEST CASE: {test_case}")
        return scenario2case_map[test_case]

    def data(self, dataset):
        data = Data_Preprocessing(dataset)
        preprocess_function = data.dataset2func_map(data.eval_data)
        samples = preprocess_function([cat], SAMPLE_COUNT_PER_DATASET)
        return samples

    def prompt(self, samples, scenario, category):
        data_list, prompt_list = Prompt_Composition().prompt_structure(data = samples, scenario = scenario, category = category)
        return data_list, prompt_list

    def generator(self, prompt_list):
        gpt = LLM()
        response_list = []
        for prompt in tqdm(prompt_list):
            try:
                response_list.append(gpt.openai_call(prompt))
            except Exception as e:
                print(f"prompt {prompt} threw exception {e}")
                response_list.append(f"ERROR: {e}")
        return response_list

    def evaluation(self, data_list, response_list, category, eval_tool):
        eval = EvaluationTools(eval_tool)
        evaluation_function = eval.tool2func_map(eval.eval_tool)
        flag_list = []
        for i in tqdm(range(len(response_list))):
            response = response_list[i]
            data = data_list[i]
            try:
                flag_list.append((response,evaluation_function(data, response, category)))
            except Exception as e:
                print(f"Evaluating response {response} threw error {e}")
                flag_list.append((response, False))

        return flag_list

if __name__ == '__main__':
    guard_test = Guard()

    for cat in TEST_CATEGORIES:
        cat_taxonomy = guard_test.taxonomy(cat)

        CAT_TEST_CASES = cat_taxonomy["test_cases"]
        for test_case in CAT_TEST_CASES:
            scenario = guard_test.scenario(test_case)

            eval_dataset_list = TAXONOMY_DICT[cat]['evaluation_datasets']
            for eval_ds in eval_dataset_list:
                try:
                    samples = guard_test.data(eval_ds)
                    data_list, prompt_list = guard_test.prompt(samples, scenario, cat)
                    response_list = guard_test.generator(prompt_list)

                    for eval_tool in TAXONOMY_DICT[cat]['evaluation_tool']:
                        result_list = guard_test.evaluation(data_list, response_list, cat, eval_tool)
                        result_df = [(samples[cat][i], result_list[i][0], result_list[i][1]) for i in range(len(samples[cat]))]
                        result_df = pd.DataFrame(result_df, columns =['Samples', 'Response', 'Flag'])
                        result_df.to_csv(f"../test_df/{cat}/{test_case}_{eval_ds}_{eval_tool}.csv")
                        print(Fore.GREEN + f"result : {result_df['Flag'].value_counts()}")
                        print(Fore.RESET + f"-"*100)

                except Exception as e:
                    print(e)
                    print(Fore.RESET + f"-" * 100)

