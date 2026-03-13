import json
import os
import time
import argparse
from tqdm import tqdm
import google.generativeai as genai

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQA-E model predictions using LLM-as-a-Judge (Gemini)")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file with model predictions. Expected format: list of dicts with 'question', 'ground_truth', 'prediction'")
    parser.add_argument("--output_json", type=str, default="llm_evaluation_results.json", help="Path to save the evaluation results with detailed scores.")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model version to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to evaluate (for testing).")
    return parser.parse_args()

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please run: export GEMINI_API_KEY='your_api_key_here'")
        exit(1)
    
    genai.configure(api_key=api_key)

def evaluate_with_gemini(question, gt_answer, pred_answer, model):
    prompt = f"""You are an expert evaluator for a Visual Question Answering (VQA) system with Explanations (VQA-E). 
You will be provided with a Question, the Ground Truth answer (which includes an answer and an explanation), and the Model's Predicted answer (also with an explanation).
Your task is to score the Model's Predicted answer on a scale of 0 to 5 based on how well it predicts the correct answer AND captures the core reasoning of the Ground Truth.
Ignore minor grammatical differences, punctuation, or tokenization artifacts (like <unk>). Focus on the semantic correctness of the answer and the reasoning.

Question: {question}
Ground Truth: {gt_answer}
Model Prediction: {pred_answer}

Return ONLY a single integer from 0 to 5 representing the score. Do not include any other text, explanation, or markdown formatting.
0: Completely wrong answer and reasoning.
1: Captures a small fragment of the truth but is mostly incorrect.
2: Partially correct but misses key reasoning or details.
3: Mostly correct meaning, but the reasoning is slightly flawed or misses a minor detail.
4: Correct meaning and reasoning, but phrased differently or less comprehensively.
5: Perfect match in meaning and reasoning.
"""
    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        # Handle cases where the model might output something like "Score: 5" or "5."
        # Keep only digits
        score_digits = ''.join(filter(str.isdigit, score_text))
        if score_digits:
            score = int(score_digits[0]) # take the first digit found
            return min(max(score, 0), 5) # Ensure it's between 0 and 5
        else:
             return 0
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None # Return None to indicate failure (maybe rate limit)

def main():
    args = parse_args()
    setup_gemini()
    
    # Initialize the model
    # Use flash for faster evaluation
    model = genai.GenerativeModel(args.model)
    
    print(f"Loading predictions from {args.input_json}")
    with open(args.input_json, 'r') as f:
        data = json.load(f)
        
    if args.limit:
        data = data[:args.limit]
        print(f"Limiting to {args.limit} samples.")
        
    results = []
    total_score = 0
    failed_calls = 0
    
    print(f"Starting LLM-as-a-Judge evaluation using {args.model}...")
    for idx, item in enumerate(tqdm(data)):
        q = item.get('question', '')
        gt = item.get('ground_truth', '')
        pred = item.get('prediction', '')
        
        score = evaluate_with_gemini(q, gt, pred, model)
        
        # Simple backoff for rate limits
        if score is None:
            failed_calls += 1
            print(f"Call failed for index {idx}. Sleeping for 10 seconds before retrying...")
            time.sleep(10)
            score = evaluate_with_gemini(q, gt, pred, model)
            if score is None:
                 print(f"Retry failed. Assigning score 0.")
                 score = 0
                 
        total_score += score
        item['llm_score'] = score
        results.append(item)
        
        # Sleep to respect rate limits (e.g. 15 RPM for free tier -> 4s sleep)
        time.sleep(4)
        
    avg_score = total_score / len(results) if results else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"Total samples evaluated: {len(results)}")
    print(f"Failed API calls: {failed_calls}")
    print(f"Average LLM Score (0-5): {avg_score:.2f}")
    
    print(f"\nSaving detailed results to {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump({
            "summary": {
                "total_samples": len(results),
                "average_score": avg_score,
                "model_used": args.model
            },
            "results": results
        }, f, indent=4)

if __name__ == "__main__":
    main()
