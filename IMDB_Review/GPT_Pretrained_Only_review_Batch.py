import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")  # Set padding_side to 'left'
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

# Load original review data
df = pd.read_csv("IMDB Dataset.csv")
review_texts = df['review'].tolist()

# Define batch size
batch_size = 8

# Generate fake review text in batches
generated_reviews = []
for i in tqdm(range(0, len(review_texts), batch_size), desc="Generating Fake Reviews"):
    batch_reviews = review_texts[i:i+batch_size]  # Ensure batch_reviews is a list of strings
    batch_reviews = [str(review) for review in batch_reviews]  # Convert each review to string
    #print("Batch reviews type:", type(batch_reviews))
    #print("Batch reviews:", batch_reviews)
    input_ids = tokenizer(batch_reviews, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    print("Input IDs type:", type(input_ids))
    attention_mask = input_ids.ne(tokenizer.pad_token_id)  # Create attention mask
    max_length = min(max(len(review) for review in batch_reviews) + 50, model.config.max_position_embeddings)
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7, attention_mask=attention_mask)
    batch_fake_reviews = [tokenizer.decode(output[0], skip_special_tokens=True) for output in outputs]
    generated_reviews.extend(batch_fake_reviews)


# Print and save generated fake reviews


# If you want to save the generated fake reviews to a CSV file
df_fake_reviews = pd.DataFrame({'Original Review': review_texts, 'Generated Fake Review': generated_reviews})
df_fake_reviews.to_csv("GPT2_Pretrained_only_review_batch.csv", index=False)

