import os
from dotenv import load_dotenv
# Import the dataset building phase
from dataset_builder import build_dataset
# Import the feature analysis phase
from feature_analyzer import analyze_features
# Import the live benchmarking phase
from live_benchmarker import run_benchmarking

# Import the report generator phase
from report_generator import generate_pdf

# Define the main orchestrator function
# Initialize environment
load_dotenv()

def main():
    # Print initialization message
    print("Initializing PromptBench Modular System...")
    
    # Run Phase 1: Build the dataset and get the DataFrame and DB reference
    df, db = build_dataset()
    
    # Run Phase 2: Analyze the features in the generated dataset
    processed_df = analyze_features(df)
    
    # Run Phase 3: Benchmark prompts against Gemini API and store results
    run_benchmarking(db)
    
    # Run Phase 4: Generate the Final PDF Report
    generate_pdf()
    
    # Print completion message
    print("\nPromptBench execution complete!")

# Check if script is run directly
if __name__ == "__main__":
    # Call main function
    main()
