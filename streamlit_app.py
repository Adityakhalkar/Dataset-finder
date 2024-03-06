import streamlit as st
import pandas as pd
from transformers import pipeline


# Load the zero-shot classification model
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# Sample dataset (replace this with your actual dataset)
df = pd.read_csv('datasets.csv')

# Function to find tags and relevant datasets
@st.cache
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
            st.success("Relevant tags:")
            for tag in relevant_tags:
                st.write(tag)

            # Display datasets corresponding to relevant tags
            st.subheader("Datasets:")
            for dataset in relevant_datasets:
                st.write(dataset)
        else:
            st.warning("No relevant tags found.")

# Run the app
if __name__ == "__main__":
    main()
