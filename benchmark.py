#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().system('pip install -qqq Levenshtein')


# In[17]:


import pandas as pd
import time
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from os import environ
from typing import Optional
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()


# In[18]:


class LLMClient:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",  # or "deepseek-chat"
        temperature: float = 0.0,
        max_tokens: int = 30,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_template: str = "{description}"  # prompt template with {description} placeholder
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template  # preset prompt template

        # Determine API key and base URL
        self.api_key = api_key 
        self.base_url = base_url or "https://api.openai.com/v1"

        # Initialize the OpenAI client with specified base URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, description: str) -> str:
        # Format the prompt using the instance's prompt template
        prompt = self.prompt_template.format(description=description)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()


# In[19]:


class HuggingFaceLLM:
    """
    HF causal‐LM wrapper supporting:
      • full-precision or 4-bit quantization
      • optional QLoRA adapter
      • uses HUGGINGFACE_TOKEN for private/gated repos
    """
    def __init__(
        self,
        model_name: str,
        prompt_template: str = "{description}",
        device_map="auto",
        torch_dtype=torch.float16,
        max_new_tokens: int = 512,
        peft_adapter_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        trust_remote_code: bool = True,
        quantize_4bit: bool = False,            # set True for 4-bit, False for full-precision
    ):
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.peft_adapter_dir = peft_adapter_dir
        # now reads from HUGGINGFACE_TOKEN
        self.hf_token = hf_token or user_secrets.get_secret("HUGGINGFACE_TOKEN")

        # Prepare 4-bit config if requested
        qconfig = None
        if quantize_4bit:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load base model (4-bit or full), then optional adapter
        if peft_adapter_dir:
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=qconfig,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=self.hf_token,
                trust_remote_code=trust_remote_code,
            )
            self.model = PeftModel.from_pretrained(
                base,
                peft_adapter_dir,
                torch_dtype=torch_dtype,
                token=self.hf_token,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=qconfig,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=self.hf_token,
                trust_remote_code=trust_remote_code,
            )

        # Shared tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=trust_remote_code,
        )

    def generate(self, description: str) -> str:
        prompt = self.prompt_template.format(description=description)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        gen_ids = outputs[:, inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


# In[20]:


def benchmark_llm(model, test_df: pd.DataFrame, incorrect_log_path: str):
    total_samples = len(test_df)
    exact_matches = 0
    total_lev_distance = 0
    timings = []
    incorrect_predictions = []

    for _, row in tqdm(test_df.iterrows(), total=total_samples, desc="Benchmarking"):
        # Use the row's description from the DataFrame
        description = row['description']
        key_no_spaces = row['key'].replace(" ", "")

        start_time = time.perf_counter()
        # Pass the description to the generate function
        prediction = model.generate(description=description).strip()
        elapsed = time.perf_counter() - start_time
        timings.append(elapsed)

        if prediction == key_no_spaces:
            exact_matches += 1
        else:
            incorrect_predictions.append({
                "description": description,
                "expected": key_no_spaces,
                "predicted": prediction
            })

        total_lev_distance += levenshtein_distance(prediction, key_no_spaces)

    accuracy = exact_matches / total_samples * 100
    avg_lev_distance = total_lev_distance / total_samples
    mean_time = sum(timings) / total_samples
    median_time = sorted(timings)[total_samples // 2]

    # Logging incorrect predictions to a CSV file
    pd.DataFrame(incorrect_predictions).to_csv(incorrect_log_path, index=False)

    return pd.DataFrame([{
        "Model": model.__class__.__name__,
        "Accuracy": accuracy,
        "Avg_Levenshtein": avg_lev_distance,
        "Mean_Time": mean_time,
        "Median_Time": median_time
    }])


# In[21]:


def benchmark_multiple_llms(models: dict, test_df: pd.DataFrame):
    results_df = pd.DataFrame()

    for model_name, model in models.items():
        print(f"\nRunning benchmark for {model_name}")
        incorrect_log_path = f"logs/incorrect_predictions_{model_name}.csv"
        summary_df = benchmark_llm(model, test_df, incorrect_log_path)
        summary_df["Model"] = model_name
        results_df = pd.concat([results_df, summary_df], ignore_index=True)

    return results_df


# In[22]:


prompt_template_default = (
    "How to {description} using vim motions? "
    "Write only the symbol sequence representing the vim motion. "
    "Don't use spaces in your answer. "
    "Combination of control and <symbol> should be written as 'Ctrl+<symbol>'."
)

prompt_template_extended = ""
with open("/kaggle/input/ext-prompt/extended_prompt.txt", "r") as file:
    prompt_template_extended = file.read()


# In[ ]:


OPENAI_API_KEY = user_secrets.get_secret("openai_api")
DEEPSEEK_API_KEY =  user_secrets.get_secret("deepseek_api")

qwen_05_default_prompt = HuggingFaceLLM('Qwen/Qwen2.5-0.5B-Instruct', prompt_template=prompt_template_default)
qwen_05_extended_prompt = HuggingFaceLLM('Qwen/Qwen2.5-0.5B-Instruct', prompt_template=prompt_template_extended)
qwen_7_default_prompt = HuggingFaceLLM('Qwen/Qwen2.5-7B-Instruct', prompt_template=prompt_template_default)
qwen_7_extended_prompt = HuggingFaceLLM('Qwen/Qwen2.5-7B-Instruct', prompt_template=prompt_template_extended)

dir_05 = '/Users/timopheymazurenko/dev/vim_tutor/kaggle/working/trained_Qwen/Qwen2.5-0.5B-Instruct'
qwen_05_finutined_default_prompt = HuggingFaceLLM('Qwen/Qwen2.5-0.5B-Instruct', peft_adapter_dir=dir_05, prompt_template=prompt_template_default)
qwen_05_finetuned_extended_prompt = HuggingFaceLLM('Qwen/Qwen2.5-0.5B-Instruct', prompt_template=prompt_template_extended)
dir_7 = '/Users/timopheymazurenko/dev/vim_tutor/kaggle/working/trained_Qwen/Qwen2.5-7B-Instruct'
qwen_7_finetuned_default_prompt = HuggingFaceLLM('Qwen/Qwen2.5-7B-Instruct', peft_adapter_dir=dir_7, prompt_template=prompt_template_default)
qwen_7_finetuned_extended_prompt = HuggingFaceLLM('Qwen/Qwen2.5-7B-Instruct', prompt_template=prompt_template_extended)

OPENAI_API_KEY = environ.get("openai_api")
gpt41_default_prompt = LLMClient("gpt-4.1", api_key=OPENAI_API_KEY, prompt_template=prompt_template_default)
gpt41mini_default_prompt = LLMClient("gpt-4.1-mini", api_key=OPENAI_API_KEY, prompt_template=prompt_template_default)
gpt41nano_default_prompt = LLMClient("gpt-4.1-nano", api_key=OPENAI_API_KEY, prompt_template=prompt_template_default)
gpt41_extended_prompt = LLMClient("gpt-4.1", api_key=OPENAI_API_KEY, prompt_template=prompt_template_extended)
gpt41mini_extended_prompt = LLMClient("gpt-4.1-mini", api_key=OPENAI_API_KEY, prompt_template=prompt_template_extended)
gpt41nano_extended_prompt = LLMClient("gpt-4.1-nano", api_key=OPENAI_API_KEY, prompt_template=prompt_template_extended)

DEEPSEEK_API_KEY = environ.get("deepseek_api")
deepseek_default_prompt = LLMClient("deepseek-chat", base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY, prompt_template=prompt_template_default)
deepseek_extended_prompt = LLMClient("deepseek-chat", base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY, prompt_template=prompt_template_extended)

llm_instances = {
    "qwen_0.5B_default_prompt": qwen_05_default_prompt,
    "qwen_0.5B_extended_prompt": qwen_05_extended_prompt,
    "qwen_7B_default_prompt": qwen_7_default_prompt,
    "qwen_7B_extended_prompt": qwen_7_extended_prompt,
    "qwen_05_finutined_default_prompt": qwen_05_finutined_default_prompt,
    "qwen_05_finetuned_extended_prompt": qwen_05_finetuned_extended_prompt,
    "qwen_7_finetuned_default_prompt": qwen_7_finetuned_default_prompt,
    "qwen_7_finetuned_extended_prompt": qwen_7_finetuned_extended_prompt, 
    "deepseek_v3_default_prompt": deepseek_default_prompt,
    "gpt_4.1_default_prompt": gpt41_default_prompt,
    "gpt_4.1_mini_default_prompt": gpt41mini_default_prompt,
    "gpt_4.1_nano_default_prompt": gpt41nano_default_prompt,
    "gpt_4.1_extended_prompt": gpt41_extended_prompt,
    "gpt_4.1_mini_extended_prompt": gpt41mini_extended_prompt,
    "gpt_4.1_nano_extended_prompt": gpt41nano_extended_prompt,
    "deepseek_v3_extended_prompt": deepseek_extended_prompt
}


# In[ ]:


for llm in llm_instances:
    llm_instances[llm].generate('Hi')

