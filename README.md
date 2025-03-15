# AI Content Moderation Framework

## Overview
This project implements an AI-driven content moderation framework that processes input data, generates responses using an LLM, and evaluates outputs against predefined safety policies. The framework supports various categories such as hate speech, sexual content, violence, and misinformation, ensuring safe and responsible AI responses.

## Features
- **Data Preprocessing:** Extracts and processes samples from multiple datasets.
- **Prompt Composition:** Generates structured prompts based on scenarios.
- **Response Generation:** Calls OpenAI's GPT models to generate responses.
- **Evaluation Tools:** Uses OpenAI moderation API and Purple Llama for content safety assessment.
- **Taxonomy-Based Categorization:** Classifies content into predefined categories for moderation.

## Project Structure
```
├── config.py               # Configuration settings for the model and evaluation categories
├── data.py                 # Data preprocessing for different datasets
├── evaluation.py           # Evaluation tools to assess response safety
├── generator.py            # GPT-based response generator
├── main_guard.py           # Main execution pipeline for content moderation
├── prompt_composition.py   # Handles prompt structuring based on scenarios
├── scenario.py             # Defines test scenarios
├── taxonomy.py             # Defines category-based content moderation rules
```

## Setup Instructions
1. **Install Dependencies:**
   ```bash
   pip install openai pandas jsonlines tqdm colorama tenacity
   ```

2. **Configure API Keys:**
   - Update `config.py` with your OpenAI and Purple Llama API keys.

3. **Run the Content Moderation Pipeline:**
   ```bash
   python main_guard.py
   ```

## How It Works
1. **Data Preprocessing:** Extracts data samples based on defined categories.
2. **Scenario-Based Prompting:** Constructs prompts tailored to different test cases.
3. **Response Generation:** Uses GPT to generate responses for each prompt.
4. **Evaluation:** Assesses the generated responses using moderation APIs.
5. **Logging & Analysis:** Saves results in structured formats for further analysis.

## Customization
- Modify `taxonomy.py` to adjust category definitions and evaluation datasets.
- Extend `scenario.py` to add new test cases.
- Change `generator.py` to use a different LLM model.

## Contact
- You can contact me on vinayakgoyaliitd26@gmail.com

