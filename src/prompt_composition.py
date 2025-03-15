
class Prompt_Composition():

    def __init__(self):
        print("prompt templatization starts")
        pass

    def prompt_structure(self, data, scenario, category):
        prompt_list = []
        prompt_list_user = [scenario[i]["content"].replace('{prompt}', f'{prompt}') for i in range(len(scenario)) if scenario[i]["role"] == "user" for prompt in data[category]]
        prompt_list_system = [scenario[i]["content"].replace('{prompt}', f'{prompt}') for i in range(len(scenario)) if scenario[i]["role"] == "system" for prompt in data[category]]
        for i in range(len(data[category])):
            prompt_list.append([
                {"role": "system", "content": prompt_list_system[i]},
                {"role": "user", "content": prompt_list_user[i]}
            ]
            )
        data_list = data[category]
        return data_list, prompt_list
