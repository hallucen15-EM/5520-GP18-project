from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datasets import load_dataset
import pandas as pd
import re

def generate_text(model, tokenizer, prompt, max_new_tokens=512, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

def create_prompts(question, answer, fewshot_examples=None):
    prompts = {}
    
    # 1. Direct Answer, zero-shot
    prompts["direct_zero"] = f"Q: {question}\nA: (Output the final numerical answer without commas, with format 'answer = ')"
    
    # 2. CoT, zero-shot
    prompts["cot_zero"] = f"Q: {question}\nA: let's think step by step: (Output the final numerical answer without commas, with format 'answer = ')"
    
    # 3. Direct Answer, fewshot
    if fewshot_examples:
        fewshot_direct = "\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in fewshot_examples[:3]])
        prompts["direct_few"] = f"{fewshot_direct}\n\nQ: {question}\nA: (Output the final numerical answer without commas, with format 'answer = ')"
    
    # 4. CoT, fewshot
    if fewshot_examples:
        fewshot_cot = "\n".join([f"Q: {ex['question']}\nA: let's think step by step:{ex['answer']}" for ex in fewshot_examples[:3]])
        prompts["cot_few"] = f"{fewshot_cot}\n\nQ: {question}\nA: let's think step by step: (Output the final numerical answer without commas, with format 'answer = ')"
    
    return prompts

def extract_final_answer(output, ground_truth):
    
    # Use re to extract final answer

    matches = re.findall(r"answer\s*=\s*(\d+(?:\s*\d+)*)", output)
    if matches:
        last_answer = matches[-1]
        return last_answer, "answer pattern"

    end_num = numbers = re.findall(r'\d+', output)
    if end_num:
        return numbers[-1], "end_num"
    
    return None, "failed"

def evaluate_gsm8k(model,tokenizer,model_path, dataset, start_idx=0, end_idx=None):    
    fewshot_examples = [
        {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell in May?", 
         "answer": "48/2 = 24, answer = 24"},
        {"question": "Lucia ate 3/5 of a bag of oranges. If she ate 21 oranges, how many oranges were in the bag originally?", 
         "answer": "21/3*5 = 35, answer = 35"}
    ]
    
    results = []
    if end_idx is None:
        end_idx = len(dataset)
    
    for i in range(start_idx, end_idx):
        question = dataset[i]["question"]
        ground_truth = dataset[i]["answer"].split("####")[-1].strip()
        
        prompts = create_prompts(question, ground_truth, fewshot_examples)
        
        for prompt_type, prompt in prompts.items():
            output = generate_text(model,tokenizer,prompt)
            pred, method = extract_final_answer(output, ground_truth)
            correct = pred == ground_truth if pred else False

            results.append({
                "id": i,
                "question": question,
                **{"prompt_type": prompt_type},
                "prompt": prompt,
                "output": output,
                "pred": pred,
                "ground_truth": ground_truth,
                "correct": correct,
                "method": method
            })
            
            print(f"{i+1}/{end_idx}: {pred} == {ground_truth} ? {correct}")
        
        print(f"done {i+1}/{end_idx}")
    
    # save
    df = pd.DataFrame(results)
    filename = f"gsm8k_cot_results[{start_idx}-{end_idx}].csv"
    df.to_csv(filename, index=False)

    print("\n=== accuracy for each prompt type ===")
    acc_by_type = df.groupby('prompt_type')['correct'].agg([
        'mean', 'count', 'sum'
    ]).round(4)
    acc_by_type['mean'] = acc_by_type['mean'].map('{:.2%}'.format)
    acc_by_type.columns = ['accu', 'total test', 'accu #']
    print(acc_by_type)

    
    # return df

def main():
    #change to the local path of your model
    local_path = "D:/5520/qwen4b/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    dataset = load_dataset("gsm8k", "main")["test"]
    print(f"in total {len(dataset)} of tests")

    # adjust the last two argument as the range to generate test result from dataset
    results_df = evaluate_gsm8k(model, tokenizer,local_path, dataset, 100,200)

if __name__ == "__main__":
    main()