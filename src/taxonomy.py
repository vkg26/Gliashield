TAXONOMY_DICT = {

    "hate":     {"definition": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
                "test_cases": ["TEST_1", "TEST_2"],
                "evaluation_datasets": ["openai_moderation_dataset", "jigsaw_wiki_dataset", "tweet_eval", "real_toxicity_prompts_dataset", "toxic_chat", "do_not_answer_dataset", "ethos_dataset_binary"],
                "evaluation_tool": ["openai_moderation_api", "purple_llama"],
                "evaluation_metrics": ["confidence_score"]
                },

    "sexual":   {
                "definition": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
                "test_cases": ["TEST_1", "TEST_2"],
                "evaluation_datasets": ["openai_moderation_dataset", "jigsaw_wiki_dataset", "real_toxicity_prompts_dataset", "toxic_chat", "do_not_answer_dataset"],
                "evaluation_tool": ["openai_moderation_api", "purple_llama"],
                "evaluation_metrics": ["confidence_score"]
                },

    "violence": {
                "definition": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
                "test_cases": ["TEST_1", "TEST_2"],
                "evaluation_datasets": ["openai_moderation_dataset", "jigsaw_wiki_dataset", "real_toxicity_prompts_dataset", "toxic_chat", "do_not_answer_dataset", "ethos_dataset_multilabel"],
                "evaluation_tool": ["openai_moderation_api", "purple_llama"],
                "evaluation_metrics": ["confidence_score"]
                },

    "harassment": {
                "definition": "Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.",
                "test_cases": ["TEST_1", "TEST_2"],
                "evaluation_datasets": ["openai_moderation_dataset", "jigsaw_wiki_dataset", "tweet_eval", "real_toxicity_prompts_dataset", "toxic_chat", "do_not_answer_dataset"],
                "evaluation_tool": ["openai_moderation_api", "purple_llama"],
                "evaluation_metrics": ["confidence_score"]
                },

    "self-harm": {
                "definition": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders",
                "test_cases": ["TEST_1", "TEST_2"],
                "evaluation_datasets": ["openai_moderation_dataset", "toxic_chat", "do_not_answer_dataset"],
                "evaluation_tool": ["openai_moderation_api", "purple_llama"],
                "evaluation_metrics": ["confidence_score"]
                },

    "toxic": {
        "definition": "content that promotes toxicity",
        "test_cases": ["TEST_1", "TEST_2"],
        "evaluation_datasets": ["real_toxicity_prompts_dataset", "jigsaw_wiki_dataset", "do_not_answer_dataset"],
        "evaluation_tool": ["purple_llama"],
        "evaluation_metrics": ["confidence_score"]
            },

    "weapons&criminal-planning": {
        "definition": "Content that promotes or contain information about the procurement or use of illegal weapons or contain information about the planning and strategizing of illegal criminal activities",
        "test_cases": ["TEST_1", "TEST_2"],
        "evaluation_datasets": ["do_not_answer_dataset"],
        "evaluation_tool": ["purple_llama"],
        "evaluation_metrics": ["confidence_score"]
                },

    "substance-abuse": {
        "definition": "Content that promotes or contain information about the procurement or use of illegal drugs/substances",
        "test_cases": ["TEST_1", "TEST_2"],
        "evaluation_datasets": ["do_not_answer_dataset"],
        "evaluation_tool": ["purple_llama"],
        "evaluation_metrics": ["confidence_score"]
                        },

    "misinformation-harms": {
        "definition": "Content that promotes fake information for false motives",
        "test_cases": ["TEST_1", "TEST_2"],
        "evaluation_datasets": ["do_not_answer_dataset"],
        "evaluation_tool": ["purple_llama"],
        "evaluation_metrics": ["confidence_score"]
                            },

    "discrimination": {
        "definition": "Content that expresses, incites, or promotes discrimination based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "test_cases": ["TEST_1", "TEST_2"],
        "evaluation_datasets": ["crows_pairs", "winogender", "discrim_eval",
                                "bbq", "do_not_answer_dataset", "ethos_dataset_multilabel"],
        "evaluation_tool": ["purple_llama"],
        "evaluation_metrics": ["confidence_score"]
                        },

    }