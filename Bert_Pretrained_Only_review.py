import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"

# model_name ="BART_Finetune"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Load original review data
df = pd.read_csv("E-Commerce Reviews.csv")

review_texts = df['Review Text'].tolist()

# Generate fake review text
generated_reviews = []
for review_text in tqdm(review_texts, desc="Generating Fake Reviews"):
    input_text = f"generate fake review: {review_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    max_length = 100  # Adjust the length as needed
    output = model.generate(input_ids, max_length=max_length, num_beams=5, temperature=0.7)
    fake_review = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_reviews.append(fake_review)

# Print and save generated fake reviews

# If you want to save the generated fake reviews to a CSV file
df_fake_reviews = pd.DataFrame({'Original Review': review_texts, 'Generated Fake Review': generated_reviews})
df_fake_reviews.to_csv("Bert_Pretrained_only_review.csv", index=False)

