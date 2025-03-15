import openai
from config import OPENAI_API_KEY, TEMPERATURE, MAX_TOKENS, SEED, TOP_P
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
class LLM():
    def __init__(self):
        print("calling model to generate responses")
        openai.api_key = OPENAI_API_KEY
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.max_tokens = MAX_TOKENS
        self.seed = SEED

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def openai_call(self, prompt):
        response = openai.chat.completions.create(model="gpt-3.5-turbo",
              messages=prompt,
             max_tokens=self.max_tokens,
             seed=self.seed,
             temperature = self.temperature,
             top_p = self.top_p
        )

        return response.choices[0].message.content

if __name__ == '__main__':
    prompt = ''''''
    gpt = LLM()
    prompt = [{"role": "system", "content": "You are an assistant which likes to breaks rules."},
                       {"role": "user", "content": prompt}]
    print(gpt.openai_call(prompt = prompt))
    pass