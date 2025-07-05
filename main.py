!pip install google-generativeai evaluate rouge-score



import time
import re
import evaluate
import google.generativeai as genai
from rouge_score import rouge_scorer



API_KEY = "YOUR_API_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')



def generate_title_and_description(script):
    prompt = f"""
    You are a YouTube SEO expert. Based on the script below, generate:
    - A catchy YouTube title (under 70 characters)
    - A concise SEO-optimized YouTube video description (strictly 25–40 words)
    - Include at least 3 relevant keywords.

    Script:
    {script}
    """
    response = model.generate_content(prompt)
    return response.text.strip()



def parse_output(output):
    title_match = re.search(r"\*\*.*Title:?\*\*\s*(.+)", output)
    desc_match = re.search(r"\*\*.*Description:?\*\*\s*(.+)", output, re.DOTALL)

    title = title_match.group(1).strip() if title_match else ""
    description = desc_match.group(1).strip() if desc_match else ""
    return title, description



test_data = [
    {
        "input": "Learn how to grow tomatoes at home using organic compost and natural fertilizers.",
        "expected_keywords": ["tomatoes", "grow", "organic", "home"],
        "reference": "Learn to grow organic tomatoes at home using compost."
    },
    {
        "input": "Discover the best AI productivity tools for creators in 2024.",
        "expected_keywords": ["AI", "productivity", "tools", "2024"],
        "reference": "Top AI tools that help creators be more productive in 2024."
    },
    {
        "input": "Flim flam foo bloop bap",
        "expected_keywords": [],
        "reference": "Explore the fun and quirky phrase flim flam foo bloop bap."
    }
]



def tokenize_text(text):
    return re.findall(r'\b\w+\b', text.lower())

def compute_metrics(pred, expected_keywords):
    pred_tokens = tokenize_text(pred)
    expected_tokens = [k.lower() for k in expected_keywords]
    matched_keywords = sum(1 for kw in expected_tokens if kw in pred_tokens)
    precision = matched_keywords / len(pred_tokens) if pred_tokens else 0
    recall = matched_keywords / len(expected_keywords) if expected_keywords else 1
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1

def compute_rouge(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 2),
        "rougeL": round(scores["rougeL"].fmeasure, 2)
    }



print("UNIT TESTING")
for sample in test_data:
    try:
        output = generate_title_and_description(sample["input"])
        assert isinstance(output, str) and len(output) > 0
        print(f"Passed for input: {sample['input'][:40]}...")
    except Exception as e:
        print(f"Failed: {str(e)}")



print("\nINTEGRATION TESTING")
try:
    start = time.time()
    output = generate_title_and_description(test_data[0]["input"])
    duration = round(time.time() - start, 2)
    print("Model successfully generated output.")
    print(f"Response Time: {duration} seconds")
except Exception as e:
    print(f"Integration Failed: {str(e)}")



print("\nUSER TESTING SIMULATION")
user_feedback_log = []
feedback_examples = [
    "Very useful and clear title.",
    "SEO description looks good.",
    "Slightly generic description."
]

for i, sample in enumerate(test_data):
    output = generate_title_and_description(sample["input"])
    user_feedback_log.append({
        "input": sample["input"],
        "output": output,
        "feedback": feedback_examples[i]
    })

for entry in user_feedback_log:
    print(f"\nInput: {entry['input']}")
    print(f"Output:\n{entry['output']}\n")
    print(f"User Feedback: {entry['feedback']}")



print("\nPERFORMANCE & NLP METRICS")
evaluation_results = []

for sample in test_data:
    input_text = sample["input"]
    expected_keywords = sample["expected_keywords"]
    reference = sample["reference"]

    start_time = time.time()
    raw_output = generate_title_and_description(input_text)
    title, description = parse_output(raw_output)
    full_pred = f"{title} {description}"
    end_time = time.time()

    precision, recall, f1 = compute_metrics(full_pred, expected_keywords)
    rouge_scores = compute_rouge(full_pred, reference)

    evaluation_results.append({
        "input": input_text,
        "output": raw_output,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "rouge1": rouge_scores["rouge1"],
        "rougeL": rouge_scores["rougeL"],
        "response_time": round(end_time - start_time, 2)
    })



print("\nFINAL EVALUATION REPORT")
for i, r in enumerate(evaluation_results):
    print(f"\n--- Sample {i+1} ---")
    print(f"Input: {r['input']}")
    print(f"Output: {r['output']}")
    print(f"Precision: {r['precision']}")
    print(f"Recall: {r['recall']}")
    print(f"F1 Score: {r['f1_score']}")
    print(f"ROUGE-1: {r['rouge1']}")
    print(f"ROUGE-L: {r['rougeL']}")
    print(f"Response Time: {r['response_time']}s")



def user_interaction_loop():
    print("Welcome to the YouTube Title & Description Generator!\n")

    while True:
        script = input("Enter your video script or summary:\n")
        if not script.strip():
            print("Please enter a valid script.\n")
            continue

        print("\nGenerating title and description...")
        output = generate_title_and_description(script)
        print("\nGenerated Output:\n")
        print(output)

        another = input("\nDo you want to generate for another script? (yes/no): ").lower().strip()
        if another != 'yes':
            print("\nThank you for using the YouTube Title & Description Generator. Goodbye!")
            break



user_interaction_loop()