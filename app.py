import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Print API key status for debugging
if not openai_api_key:
    print("WARNING: No OpenAI API key found in environment variables")
else:
    print(f"API key found (starts with: {openai_api_key[:4]}...)")

# Define the RecipeState for Streamlit
class RecipeState:
    def __init__(self):
        self.ingredients = []
        self.dietary_restrictions = []
        self.preferences = []
        self.generated_recipe = None
        self.adjusted_recipe = None
        self.substitutions = None
        self.favorites = []
        self.user_notes = None

state = RecipeState()

# User input section
def user_input():
    st.title("SmartChef AI")

    # User inputs
    ingredients_input = st.text_input("Enter ingredients separated by commas")
    restrictions_input = st.text_input("Enter dietary restrictions separated by commas (or leave blank)")
    preferences_input = st.text_input("Enter taste or cuisine preferences separated by commas")

    # Update state with user inputs
    if ingredients_input:
        state.ingredients = [item.strip() for item in ingredients_input.split(",") if item.strip()]
    if restrictions_input:
        state.dietary_restrictions = [item.strip() for item in restrictions_input.split(",") if item.strip()]
    if preferences_input:
        state.preferences = [item.strip() for item in preferences_input.split(",") if item.strip()]

    if st.button("Generate Recipe"):
        generate_recipe()

# Recipe generation section
def generate_recipe():
    if not state.ingredients:
        st.error("No ingredients provided. Cannot generate recipe.")
        return
    
    prompt = f"""You are a helpful chef. Create a recipe with the following ingredients: {', '.join(state.ingredients)}.
    Make sure the recipe follows these dietary restrictions: {', '.join(state.dietary_restrictions) if state.dietary_restrictions else 'none'} and 
    try to match these taste preferences: {', '.join(state.preferences) if state.preferences else 'none'}
    
    Provide the recipe name, ingredients list and step-by-step instructions.
    """

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            api_key=openai_api_key
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state.generated_recipe = response.content
        st.subheader("Generated Recipe")
        st.write(state.generated_recipe)

        st.button("Adjust Recipe", on_click=adjust_recipe)
        st.button("Suggest Substitutions", on_click=suggest_substitutions)

    except Exception as e:
        st.error(f"Error generating recipe: {str(e)}")

# Diet adjustment section
def adjust_recipe():
    if not state.generated_recipe:
        st.error("No recipe to adjust")
        return
    
    prompt = f"""The following recipe may not fully comply with these dietary restrictions: {', '.join(state.dietary_restrictions)}.

    Recipe:
    {state.generated_recipe}

    Please adjust the recipe to strictly follow these dietary restrictions and make minimal necessary substitutions.
    Output the adjusted recipe in the same format.
    """

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            api_key=openai_api_key
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        state.adjusted_recipe = response.content
        st.subheader("Adjusted Recipe")
        st.write(state.adjusted_recipe)

    except Exception as e:
        st.error(f"Error adjusting recipe: {str(e)}")

# Ingredient substitution section
def suggest_substitutions():
    if not state.generated_recipe and not state.adjusted_recipe:
        st.error("No valid recipe to suggest substitutions for")
        return

    recipe_text = state.adjusted_recipe if state.adjusted_recipe else state.generated_recipe
    prompt = f"""
    Analyze the following recipe and suggest ingredient substitutions that:
    - Align with these dietary restrictions: {', '.join(state.dietary_restrictions) or 'none'}
    - Match these preferences: {', '.join(state.preferences) or 'none'}

    Recipe:
    {recipe_text}

    Return ONLY a JSON dictionary of substitutions like: {{"ingredient": "substitute"}}
    No additional text before or after the JSON.
    """

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            api_key=openai_api_key
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        substitutions = json.loads(response.content.strip())
        
        st.subheader("Suggested Substitutions")
        for ingredient, substitute in substitutions.items():
            st.write(f"{ingredient}: {substitute}")

    except Exception as e:
        st.error(f"Error suggesting substitutions: {str(e)}")

# Feedback collection section
def collect_feedback():
    feedback = st.text_input("Did you face any difficulty making the recipe? (Describe or leave blank)")
    if st.button("Submit Feedback"):
        state.user_notes = feedback
        st.write(f"Feedback submitted: {feedback}")

# Save to favorites section
def save_to_favorites():
    if state.adjusted_recipe or state.generated_recipe:
        save = st.radio("Do you want to save this recipe to favorites?", ("Yes", "No"))
        if save == "Yes":
            if state.adjusted_recipe:
                state.favorites.append(state.adjusted_recipe)
            else:
                state.favorites.append(state.generated_recipe)
            st.write("Recipe saved to favorites!")
        notes = st.text_area("Any personal notes you'd like to save with it?", height=100)
        if notes:
            state.user_notes = notes
            st.write(f"Note saved: {notes}")
    else:
        st.error("No recipe to save")

# Main Streamlit app workflow
def main():
    user_input()
    if state.generated_recipe:
        collect_feedback()
        save_to_favorites()

if __name__ == "__main__":
    main()
