from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import torch

def generate_full_word(input_ids, model, tokenizer, threshold=0.95):
    generated_ids = input_ids
    while True:
        outputs = model(generated_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(next_token_probs, dim=-1)
        next_token_prob = next_token_probs[0, next_token_id]

        if next_token_prob < threshold:
            break

        generated_ids = torch.cat((generated_ids, next_token_id.unsqueeze(0)), dim=1)

    return generated_ids

def get_mnli_label(sentence_1, sentence_2, model, tokenizer):
    inputs = tokenizer(sentence_1, sentence_2, return_tensors="pt")
    # make a prediction
    outputs = model(**inputs)
    # get the predicted class
    predicted_class_idx = outputs.logits.argmax().item()
    # get the predicted class name
    predicted_class_name = model.config.id2label[predicted_class_idx]
    return predicted_class_name

def evaluate_hallucination(sentence, base_tokenizer, base_model, mnli_model, mnli_tokenizer):
    input = sentence['model_input'] + sentence['model_output_text']
    input_words = input.split(" ")
    words_to_skip = len(sentence['model_input'].split(" "))

    labels = []


    print(f"Full sentence: {' '.join(input_words)}")
    print("\n") 

    for i, sample_to_evaluate in enumerate(input_words):
        if i == len(input_words) - 1 or i < words_to_skip-1:
            continue

        positive_influence = 0
        total_influence = 0
        # print the actual token and the next one generated 
        print(f"Actual token: {sample_to_evaluate}")
        next_generated_token = input_words[i+1]
        print(f"Next generated token: {next_generated_token}")

        sentence_until_now = " ".join(input_words[:i+1])
        print(f"Sentence until now: {sentence_until_now}")
        token_id_until_now = base_tokenizer.encode(sentence_until_now)
        token_id_until_now = torch.tensor(token_id_until_now).to(base_model.device).unsqueeze(0)
        len_token_id_until_now = token_id_until_now.shape[1]

        # evaluate top k tokens for the next word after until_now
        probabilities = base_model(token_id_until_now, return_dict=True).logits.softmax(dim=-1)
        probabilities = probabilities[:, -1, :]
        topk_probabilities, topk_indices = probabilities.topk(10, dim=-1)

        for j in range(10): 
            token_id = topk_indices[0][j].item()
            token_prob = topk_probabilities[0][j].item()
            if token_prob < 0.01:
                break
            topk_token_ids = generate_full_word(torch.cat((token_id_until_now, torch.tensor([[token_id]]).to(base_model.device)), dim=1), base_model, base_tokenizer)
            token = base_tokenizer.decode(topk_token_ids[0][len_token_id_until_now:], skip_special_tokens=True)
            if " " in token:
                token = token.split(" ")[1]

            relateness = get_mnli_label(next_generated_token, token, mnli_model, mnli_tokenizer)
            if relateness == "entailment":
                positive_influence += token_prob
                total_influence += token_prob
            elif relateness == "contradiction":
                total_influence += token_prob
            print(f"Token: {token}, Relateness: {relateness}, Probability: {token_prob}")
            torch.cuda.empty_cache()
        
        if total_influence == 0:
            hallucination_score = 0
        else:
            hallucination_score = 1 - (positive_influence/total_influence)
        labels.append(hallucination_score)
        print(f"Hallucination Score: {hallucination_score}")
        print("\n")
    return labels

if __name__ == "__main__":
    # Load the model and tokenizer
    model_path = "Qwen/QwQ-32B-Preview"
    base_model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit = True, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    mnli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-xsmall")
    mnli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-xsmall")

    # only cuda visible devices 1 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    data_dir = "data/test"
    data_file = "mushroom.en-tst.v1.jsonl"
    data_path = os.path.join(data_dir, data_file)

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    print(len(data))
    
    # check if the sentence was already processed
    output_path = os.path.join(data_dir, "results.jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            processed_data = [json.loads(line) for line in f]
        processed_ids = [sample['id'] for sample in processed_data]
        data = [sample for sample in data if sample['id'] not in processed_ids]
    
    print(len(data))
    
    for i in range(len(data)):
        sentence = data[i]
        try:
            print(f"Processing sample {i}")
            labels = evaluate_hallucination_cache(sentence, tokenizer, base_model, mnli_model, mnli_tokenizer)
            sentence['hallucination_scores_evaluated'] = labels

            output_path = os.path.join(data_dir, "results.jsonl")
            with open(output_path, "a") as f:
                f.write(json.dumps(sentence) + "\n")
        except RuntimeError as e:
            try: 
                print(f"Processing sample {i}")
                labels = evaluate_hallucination(sentence, tokenizer, base_model, mnli_model, mnli_tokenizer)
                sentence['hallucination_scores_evaluated'] = labels

                output_path = os.path.join(data_dir, "results.jsonl")
                with open(output_path, "a") as f:
                    f.write(json.dumps(sentence) + "\n")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory error for sample {i}, skipping to the next sample.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
# import json
# import torch
# import copy

def generate_full_word_cache(input_ids, model, tokenizer, past_key_values=None, threshold=0.95):
    generated_ids = input_ids
    past_values = past_key_values
    while True:
        outputs = model(generated_ids, past_key_values=past_values, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_values = outputs.past_key_values
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(next_token_probs, dim=-1)
        next_token_prob = next_token_probs[0, next_token_id]

        if next_token_prob < threshold:
            break

        generated_ids = torch.cat((generated_ids, next_token_id.unsqueeze(0)), dim=1)
        torch.cuda.empty_cache()

    return generated_ids

def evaluate_hallucination_cache(sentence, base_tokenizer, base_model, mnli_model, mnli_tokenizer):
    input = sentence['model_input'] + sentence['model_output_text']
    input_words = input.split(" ")
    words_to_skip = len(sentence['model_input'].split(" "))

    labels = []

    print(f"Full sentence: {' '.join(input_words)}")
    print("\n") 

    past_key_values = None

    for i, sample_to_evaluate in enumerate(input_words):
        if i == len(input_words) - 1 or i < words_to_skip-1:
            continue

        positive_influence = 0
        total_influence = 0
        print(f"Actual token: {sample_to_evaluate}")
        next_generated_token = input_words[i+1]
        print(f"Next generated token: {next_generated_token}")

        sentence_until_now = " ".join(input_words[:i+1])
        print(f"Sentence until now: {sentence_until_now}")
        token_id_until_now = base_tokenizer.encode(sentence_until_now)
        token_id_until_now = torch.tensor(token_id_until_now).to(base_model.device).unsqueeze(0)
        len_token_id_until_now = token_id_until_now.shape[1]

        outputs = base_model(token_id_until_now, past_key_values=past_key_values, return_dict=True, use_cache=True)
        probabilities = outputs.logits.softmax(dim=-1)
        probabilities = probabilities[:, -1, :]
        past_key_values = outputs.past_key_values
        topk_probabilities, topk_indices = probabilities.topk(10, dim=-1)

        for j in range(10): 
            token_id = topk_indices[0][j].item()
            token_prob = topk_probabilities[0][j].item()
            if token_prob < 0.01:
                break
            topk_token_ids = generate_full_word(torch.cat((token_id_until_now, torch.tensor([[token_id]]).to(base_model.device)), dim=1), base_model, base_tokenizer, past_key_values=past_key_values)
            token = base_tokenizer.decode(topk_token_ids[0][len_token_id_until_now:], skip_special_tokens=True)
            if " " in token:
                token = token.split(" ")[1]

            relateness = get_mnli_label(next_generated_token, token, mnli_model, mnli_tokenizer)
            if relateness == "entailment":
                positive_influence += token_prob
                total_influence += token_prob
            elif relateness == "contradiction":
                total_influence += token_prob
            print(f"Token: {token}, Relateness: {relateness}, Probability: {token_prob}")
            torch.cuda.empty_cache()
        
        if total_influence == 0:
            hallucination_score = 0
        else:
            hallucination_score = 1 - (positive_influence/total_influence)
        labels.append(hallucination_score)
        print(f"Hallucination Score: {hallucination_score}")
        print("\n")
    return labels

# if __name__ == "__main__":
#     model_path = "Qwen/QwQ-32B-Preview"
#     base_model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map='cuda')
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     mnli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-xsmall")
#     mnli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-xsmall")

#     data_dir = "data/test"
#     data_file = "mushroom.en-tst.v1.jsonl"
#     data_path = os.path.join(data_dir, data_file)

#     with open(data_path, "r") as f:
#         data = [json.loads(line) for line in f]

#     print(len(data))
    
#     output_path = os.path.join(data_dir, "results.jsonl")
#     if os.path.exists(output_path):
#         with open(output_path, "r") as f:
#             processed_data = [json.loads(line) for line in f]
#         processed_ids = [sample['id'] for sample in processed_data]
#         data = [sample for sample in data if sample['id'] not in processed_ids]
    
#     print(len(data))
    
#     for i in range(len(data)):
#         sentence = data[i]
#         try:
#             print(f"Processing sample {i}")
#             labels = evaluate_hallucination(sentence, tokenizer, base_model, mnli_model, mnli_tokenizer)
#             sentence['hallucination_scores_evaluated'] = labels

#             output_path = os.path.join(data_dir, "results.jsonl")
#             with open(output_path, "a") as f:
#                 f.write(json.dumps(sentence) + "\n")
#         except RuntimeError as e:
#             if "CUDA out of memory" in str(e):
#                 print(f"CUDA out of memory error for sample {i}, skipping to the next sample.")
#                 torch.cuda.empty_cache()
#                 continue
#             else:
#                 raise e