import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the zero-shot classification model
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# Sample dataset (replace this with your actual dataset)
df = pd.read_csv('/content/Dataset-finder/datasets.csv')

def tag_finder(user_input):
    keywords = df['Keyword'].unique()
    result = classifier(user_input, keywords)
    threshold = result['scores'][0]
    for score in result['scores']:
        if score == threshold:
            continue
        if (threshold - score) >= threshold / 10:
            threshold = score
        else:
            break
    useful_tags = [result['labels'][idx] for idx, score in enumerate(result['scores']) if score >= threshold]
    relevant_datasets = []
    for tag in useful_tags:
        relevant_datasets.extend(df[df['Keyword'] == tag]['Datasets'].tolist())
    return useful_tags, relevant_datasets

# Define the Streamlit app
def main():
    # Set title and description
    st.title("Dataset Tagging System")
    st.write("Enter your text below and get relevant tags for your dataset.")

    # Get user input
    user_input = st.text_input("Enter your text:")

    if st.button("Submit"):
        # Find relevant tags and datasets
        relevant_tags, relevant_datasets = tag_finder(user_input)
        
        # Display relevant tags
        if relevant_tags:
            st.subheader("Datasets:")
            for dataset in relevant_datasets:
                tag = df[df['Datasets'] == dataset]['Keyword'].iloc[0]
                st.markdown(f'''
                    <div style="border: 2px solid #555; border-radius: 10px; padding: 10px; margin-bottom: 10px; background-color: #333; color: white;">
                        <div>{dataset}</div>
                        <div style="border: 1px solid #666; padding: 5px; background-color: #444;">{tag}</div>
                    </div>
                ''', unsafe_allow_html=True)
        else:
            st.warning("No relevant tags found.")

# Run the app
if __name__ == "__main__":
    main()
