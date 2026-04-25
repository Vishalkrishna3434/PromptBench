# Import requests module for GitHub API calls
import requests
# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical operations
import numpy as np
# Import MongoClient for database operations
from pymongo import MongoClient

# Define function for dataset building
def build_dataset():
    # Print start label
    print("O/P: Starting Phase 1 (Dataset Building)...")
    
    # Initialize an empty list to hold scraped prompt texts
    prompt_texts = []
    
    try:
        # Fetch the real CSV dataset from the prompts.chat repository
        print("O/P: Fetching real prompt dataset from f/prompts.chat...")
        external_df = pd.read_csv("https://raw.githubusercontent.com/f/prompts.chat/main/prompts.csv")
        
        # We sample up to 100 random rows to keep analysis speed fast
        sample_size = min(100, len(external_df))
        sampled_prompts = external_df.sample(n=sample_size, random_state=42)
        
        # Extract the 'prompt' column as a list of strings
        prompt_texts = sampled_prompts['prompt'].dropna().astype(str).tolist()
        print(f"O/P: Successfully loaded {len(prompt_texts)} real prompts!")
    except Exception as e:
        # Print error and fallback
        print(f"O/P: Failed to fetch real data ({e}). Falling back to dummy generation.")

    # Initialize an empty list to store feature dictionaries
    scraped_data = []
    # Iterate through each scraped prompt text
    for text in prompt_texts:
        # Calculate the token length by splitting text by spaces
        token_length = len(text.split(" "))
        # Feature 2: Instruction Count
        instruction_verbs = ['always','never','do','don\'t','start','follow','include','end','keep','ensure','must','strictly','exactly']
        instruction_count = 0
        for line in text.lower().split('\n'):
            line = line.strip()
            if any(line.startswith(str(i)) for i in range(10)) or any(line.startswith(v) for v in instruction_verbs):
                instruction_count += 1

        # Feature 3: Specificity Score
        specificity_keywords = ["must", "only", "strictly", "require", "exactly", "specific", "detailed", "ensure", "always", "never"]
        num_constraints = sum(1 for word in text.lower().split() if word in specificity_keywords)
        specificity_score = (num_constraints / token_length * 100) if token_length > 0 else 0
        # Count occurrences of 'example' and 'e.g.' in the text
        example_count = text.lower().count("example") + text.lower().count("e.g.")
        # Count punctuation marks in the text
        num_punctuation = sum(1 for char in text if char in ".,;:!?()[]{}")
        # Calculate constraint density (avoid division by zero)
        constraint_density = (num_punctuation / token_length) if token_length > 0 else 0
        # Append the calculated features as a dictionary to our list
        scraped_data.append({
            "prompt_text": text,
            "token_length": token_length,
            "instruction_count": instruction_count,
            "specificity_score": specificity_score,
            "example_count": example_count,
            "constraint_density": constraint_density
        })

    # Convert the list of feature dictionaries into a pandas DataFrame
    df = pd.DataFrame(scraped_data)

    # If the DataFrame is empty (e.g., API rate limit), create dummy data
    if df.empty:
        # Import random module
        import random
        # Define lists for combinatorial prompt generation
        subjects = ["assistant", "data scientist", "translator", "writer", "tutor", "engineer", "reviewer", "analyst", "coach", "manager"]
        actions = ["Answer clearly.", "Provide structured answers.", "Translate this text.", "Summarize the article.", "Write a python script.", "Identify all bugs.", "Respond in JSON.", "Create a unique story.", "Adhere to facts.", "Explain step-by-step."]
        constraints = ["Always do your best.", "Strictly follow instructions.", "Only output the translation.", "Always provide a concise summary.", "It must be efficient.", "Do not ignore constraints.", "Do not include conversational text.", "Never use cliches.", "Strictly adhere to facts.", "Don't skip steps."]
        examples = ["Example: Be good.", "Example: memory leaks.", "e.g. hello -> hola.", "e.g., use list comprehensions.", "e.g., strictly JSON.", "Example: 2+2=4.", "e.g. bullet points.", "Example: avoid jargon.", "e.g., give a python example.", "Example: Be creative."]
        
        # Initialize an empty list for dummy data
        dummy_data = []
        # Loop to create 100 dummy rows
        for i in range(100):
            # Select random elements
            s = random.choice(subjects)
            a = random.choice(actions)
            c = random.choice(constraints)
            e = random.choice(examples)
            # Combine into a unique prompt text
            unique_text = f"You are a {s}. {a} {c} {e} [ID:{i}]"
            
            # Append a dictionary with random dummy values
            dummy_data.append({
                "prompt_text": unique_text,
                "token_length": np.random.randint(10, 600),
                "instruction_count": np.random.randint(0, 15),
                "specificity_score": np.random.uniform(0, 50),
                "example_count": np.random.randint(0, 5),
                "constraint_density": np.random.uniform(0, 0.2)
            })
        # Create DataFrame from the dummy data list
        df = pd.DataFrame(dummy_data)
        # Inject some NaN values to demonstrate missing data handling later
        df.loc[10:20, 'specificity_score'] = np.nan

    # Save the DataFrame to a CSV file named prompt_dataset.csv
    df.to_csv("prompt_dataset.csv", index=False)

    # Connect to local MongoDB instance
    client = MongoClient("mongodb://localhost:27017/")
    # Access the promptbench_db database
    db = client["promptbench_db"]
    # Access the scraped_prompts collection
    collection = db["scraped_prompts"]
    # Check if the DataFrame is not empty
    if not df.empty:
        # Insert the DataFrame records into the MongoDB collection
        collection.insert_many(df.to_dict("records"))
        
    # Return the generated DataFrame and the database client reference
    return df, db
