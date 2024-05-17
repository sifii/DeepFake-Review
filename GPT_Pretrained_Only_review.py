import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # If available, set the device to CUDA
    device = torch.device("cuda:0")  # You might want to change the device index if you have multiple GPUs
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(device))
else:
    # If CUDA is not available, set the device to CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Print the device being used
print("Device:", device)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can choose any variant of GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
# Load original review data
df = pd.read_csv("E-Commerce Reviews.csv")

review_texts = df['Review Text'].tolist()

# Generate fake review text
generated_reviews = []
for review_text in tqdm(review_texts, desc="Generating Fake Reviews"):
    input_ids = tokenizer.encode(review_text, return_tensors='pt').to(device)
    max_length = len(review_text) + 50  # Adjust the length as needed
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7)
    fake_review = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_reviews.append(fake_review)

# Print and save generated fake reviews
for original, fake in zip(review_texts, generated_reviews):
    print("Original Review:", original)
    print("Generated Fake Review:", fake)
    print("\n")

# If you want to save the generated fake reviews to a CSV file
df_fake_reviews = pd.DataFrame({'Original Review': review_texts, 'Generated Fake Review': generated_reviews})
df_fake_reviews.to_csv("GPT2_Pretrained_only_review.csv", index=False)

