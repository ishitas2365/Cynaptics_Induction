# finetune_for_persona
Cynaptics induction task

Link to Hugging Face repo (for model weights, visit here)- https://huggingface.co/ishitas2365/llama-3.2-3b-instruct-finetunedToPersona

# Provision-
We are given a persona-based chat dataset which consists of a hypothetical conversation between 2 personas A and B. We have to finetune any LLM on this dataset to inherit a persona and respond in a humane way.

# Choice of LLM: Meta Llama-3.2-3B-Instruct

-Since we are confined to computation constraints while using Kaggle (16GB GPU memory and 30 hours of P100 GPU usage per week), a lightweight model would do our work. 

-Meta Llama-3.2-3B-Instruct is a recent model last updated in October 2024. 

-It is a pretrained and instruction tuned auto regressive model with transformer architecture. It understands and responds to user instructions effectively.

-High MMLU score

-Roles supported by model- system, user, assistant, ipython

-Supports QLoRA

# Insights-
-I tried Microsoft-phi3 but encountered “Runtime Error: FlashAttention only supports Ampere GPUs or newer.” (Flash Attention is required to load and train this model)

-I tried Google-Gemma2-2B but this model does not support system prompts.

# Fine Tuning Method : QLoRA 
4-bit quantisation

# Example usage from Hugging Face using Transformers library
```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ishitas2365/llama-3.2-3b-instruct-finetunedToPersona")
model = AutoModelForCausalLM.from_pretrained("ishitas2365/llama-3.2-3b-instruct-finetunedToPersona")

tokenizer.pad_token_id = tokenizer.eos_token_id

# Enter the characteristics of persona in system prompt and the initial dialogue of the user in user prompt
messages = [
    {
        "role": "system",
        "content": "Persona B's characteristics: My name is David, and I'm a 35-year-old math teacher. "
                   "I like to hike and spend time in nature. I'm married with two kids."
    },
    {
        "role": "user",
        "content": "Morning! I think I saw you at the parent meeting, what's your name?"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=200,
        num_return_sequences=1,
        temperature=0.8,  
        top_p=0.9        
    )
decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "assistant" in decoded_text:
    response = decoded_text.split("assistant", 1)[1].strip()
else:
    response = decoded_text.strip()

print("Assistant's Reply:", response)
```
