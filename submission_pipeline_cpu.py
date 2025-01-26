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
    inputs = tokenizer(sentence_1, sentence_2, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax().item()
    predicted_class_name = model.config.id2label[predicted_class_idx]
    return predicted_class_name

def evaluate_hallucination(sentence, base_tokenizer, base_model, mnli_model, mnli_tokenizer):
    input = "[Question] " + sentence['model_input'] + " [Answer] " + sentence['model_output_text']
    input_words = input.split(" ")
    words_to_skip = len(sentence['model_input'].split(" ")) + 2

    labels = []

    for i, sample_to_evaluate in enumerate(input_words):
        if i == len(input_words) - 1 or i < words_to_skip-1:
            continue

        positive_influence = 0
        total_influence = 0

        sentence_until_now = " ".join(input_words[:i+1])
        token_id_until_now = base_tokenizer.encode(sentence_until_now)
        token_id_until_now = torch.tensor(token_id_until_now).unsqueeze(0)
        len_token_id_until_now = token_id_until_now.shape[1]

        probabilities = base_model(token_id_until_now, return_dict=True).logits.softmax(dim=-1)
        probabilities = probabilities[:, -1, :]
        topk_probabilities, topk_indices = probabilities.topk(10, dim=-1)

        for j in range(10): 
            token_id = topk_indices[0][j].item()
            token_prob = topk_probabilities[0][j].item()
            if token_prob < 0.01:
                break
            topk_token_ids = generate_full_word(torch.cat((token_id_until_now, torch.tensor([[token_id]])), dim=1), base_model, base_tokenizer)
            token = base_tokenizer.decode(topk_token_ids[0][len_token_id_until_now:], skip_special_tokens=True)
            if " " in token:
                token = token.split(" ")[1]

            next_generated_token = input_words[i+1]
            relateness = get_mnli_label(next_generated_token, token, mnli_model, mnli_tokenizer)
            
            if relateness == "entailment":
                positive_influence += token_prob
                total_influence += token_prob
            elif relateness == "contradiction":
                total_influence += token_prob

        if total_influence == 0:
            hallucination_score = 0
        else:
            hallucination_score = 1 - (positive_influence/total_influence)
        labels.append(hallucination_score)
    
    return labels

if __name__ == "__main__":
    # Load models with low memory configuration
    model_path = "Qwen/QwQ-32B-Preview"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,  # Reduce memory usage
        low_cpu_mem_usage=True,  # Optimize CPU memory
        device_map='cpu'
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_path)

    mnli_model = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/nli-deberta-v3-xsmall", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cpu'
    )
    mnli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-xsmall")

    data_dir = "data/test"
    data_file = "mushroom.en-tst.v1.jsonl"
    data_path = os.path.join(data_dir, data_file)

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    for i in range(len(data)):
        sentence = data[i]
        try:
            labels = evaluate_hallucination(sentence, base_tokenizer, base_model, mnli_model, mnli_tokenizer)
            sentence['hallucination_scores_evaluated'] = labels

            output_path = os.path.join(data_dir, "results.jsonl")
            with open(output_path, "a") as f:
                f.write(json.dumps(sentence) + "\n")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue