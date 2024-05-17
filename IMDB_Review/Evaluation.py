import pandas as pd

def remove_duplicates(original, fake):
    original_sentences = str(original).split('.')
    cleaned_fake = fake
    for sentence in original_sentences:
        cleaned_fake = cleaned_fake.replace(sentence.strip(), '')
    return cleaned_fake.strip()

# Apply the function to each row
df = pd.read_csv('GPT2_Pretrained_only_review_batch.csv')
df['Fake Review'] = df.apply(lambda row: remove_duplicates(row['Original Review'], row['Generated Fake Review']), axis=1)

# Display the updated DataFrame
df['Fake Review'] = df['Fake Review'].apply(lambda x: x.replace("generate fake review: ", ""))

combined_reviews = []
for original, fake in zip(df['Original Review'], df['Fake Review']):
    combined_reviews.extend([original, fake])

# Create labels
labels = [0, 1] * len(df['Fake Review'])

# Create DataFrame
df1 = pd.DataFrame({'Text': combined_reviews, 'Label': labels})


df1 = df1.sample(frac=1).reset_index(drop=True)

print(df1)



df1.to_csv('Roberta_model1.csv', index=False)
