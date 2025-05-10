from pydantic import BaseModel, Field
from typing import Dict, Optional, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage  # Fixed import
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

class RecipeState(TypedDict):
    ingredients: List[str]
    dietary_restrictions: List[str]
    preferences: List[str]
    
    generated_recipe: Optional[str]
    adjusted_recipe: Optional[str]
    substitutions: Optional[Dict[str, str]]
    
    meal_plan: Optional[Dict[str, List[str]]]
    shopping_list: Optional[List[str]]
    
    feedback: Optional[str]
    favorites: Optional[List[str]]
    user_notes: Optional[str]


def user_input_node(state: RecipeState) -> Dict:
    """Get user input for ingredients, restrictions, and preferences"""
    ingredients_input = input("Enter ingredients separated by commas: ")
    restrictions_input = input("Enter diet restrictions separated by commas (or leave blank): ")
    preferences_input = input("Enter taste or cuisine preferences separated by commas: ")

    ingredients = [item.strip() for item in ingredients_input.split(',') if item.strip()]
    dietary_restrictions = [item.strip() for item in restrictions_input.split(',') if item.strip()]
    preferences = [item.strip() for item in preferences_input.split(',') if item.strip()]

    print(f"\nIngredients: {ingredients}")
    print(f"Dietary Restrictions: {dietary_restrictions}")
    print(f"Preferences: {preferences}")

    return {
        "ingredients": ingredients,
        "dietary_restrictions": dietary_restrictions,
        "preferences": preferences
    }


def recipe_generation_node(state: RecipeState) -> Dict:
    """Generate a recipe based on user inputs"""
    ingredients = state.get("ingredients", [])
    dietary_restrictions = state.get("dietary_restrictions", [])
    preferences = state.get("preferences", [])

    if not ingredients:
        print("No ingredients provided. Cannot generate recipe.")
        return {"generated_recipe": "No ingredients provided. Cannot generate recipe."}

    prompt = f"""You are a helpful chef. Create a recipe with the following ingredients: {', '.join(ingredients)}.
    Make sure the recipe follows these dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'none'} and 
    try to match these taste preferences: {', '.join(preferences) if preferences else 'none'}
    
    Provide the recipe name, ingredients list and step-by-step instructions.
    """

    try:
        # Create a new instance of the ChatOpenAI model for each request
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            api_key=openai_api_key
        )
        
        print("\nSending recipe request to OpenAI...")
        # Send the message using the correct format
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Extract content from the response
        generated_recipe = response.content
        
        print("\n--- GENERATED RECIPE ---")
        print(generated_recipe[:200] + "..." if len(generated_recipe) > 200 else generated_recipe)
        
        return {"generated_recipe": generated_recipe}
    except Exception as e:
        error_message = f"Error generating recipe: {str(e)}"
        print(f"\nERROR: {error_message}")
        import traceback
        traceback.print_exc()
        return {"generated_recipe": error_message}


def diet_adjustment_node(state: RecipeState) -> Dict:
    """Adjust recipe to dietary restrictions if needed"""
    original_recipe = state.get("generated_recipe", "")
    dietary_restrictions = state.get("dietary_restrictions", [])

    if not original_recipe or "Error" in original_recipe or not dietary_restrictions:
        print("\nSkipping diet adjustment (no recipe or no dietary restrictions)")
        return {"adjusted_recipe": original_recipe}

    prompt = f"""The following recipe may not fully comply with these dietary restrictions: {', '.join(dietary_restrictions)}.

    Recipe:
    {original_recipe}

    Please adjust the recipe to strictly follow these dietary restrictions and make minimal necessary substitutions.
    Output the adjusted recipe in the same format.
    """

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            api_key=openai_api_key
        )
        
        print("\nAdjusting recipe for dietary restrictions...")
        response = llm.invoke([HumanMessage(content=prompt)])
        adjusted_recipe = response.content
        
        print("\n--- ADJUSTED RECIPE ---")
        print(adjusted_recipe[:200] + "..." if len(adjusted_recipe) > 200 else adjusted_recipe)
        
        return {"adjusted_recipe": adjusted_recipe}
    except Exception as e:
        error_message = f"Error adjusting recipe: {str(e)}"
        print(f"\nERROR: {error_message}")
        import traceback
        traceback.print_exc()
        return {"adjusted_recipe": original_recipe}


def ingredient_substitution_node(state: RecipeState) -> Dict:
    """Generate substitution suggestions"""
    recipe_text = state.get("adjusted_recipe") or state.get("generated_recipe", "")
    dietary_restrictions = state.get("dietary_restrictions", [])
    preferences = state.get("preferences", [])

    if not recipe_text or "Error" in recipe_text:
        print("\nSkipping substitutions (no valid recipe)")
        return {"substitutions": {}}

    prompt = f"""
    Analyze the following recipe and suggest ingredient substitutions that:
    - Align with these dietary restrictions: {', '.join(dietary_restrictions) or 'none'}
    - Match these preferences: {', '.join(preferences) or 'none'}

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
        
        print("\nGenerating ingredient substitutions...")
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Extract JSON from the response
        try:
            # Try to parse the whole string as JSON
            substitutions = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, look for JSON-like structure between braces
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    substitutions = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    substitutions = {"error": "Could not parse JSON from response"}
            else:
                substitutions = {"error": "No JSON found in response"}
        
        print("\n--- SUGGESTED SUBSTITUTIONS ---")
        for ingredient, substitute in substitutions.items():
            print(f"- {ingredient}: {substitute}")
            
        return {"substitutions": substitutions}
    except Exception as e:
        error_message = f"Failed to extract substitutions: {str(e)}"
        print(f"\nERROR: {error_message}")
        import traceback
        traceback.print_exc()
        return {"substitutions": {"error": error_message}}


def feedback_node(state: RecipeState) -> Dict:
    """Collect user feedback"""
    print("\n--- FEEDBACK ---")
    feedback = input("Did you face any difficulty making the recipe? (Describe or leave blank): ").strip()
    return {"feedback": feedback}


def storage_node(state: RecipeState) -> Dict:
    """Save recipe to favorites if desired"""
    recipe = state.get("adjusted_recipe") or state.get("generated_recipe", "")
    if not recipe or "Error" in recipe:
        print("\nNo valid recipe to save")
        return {"favorites": state.get("favorites", []), "user_notes": ""}

    print("\n--- SAVE RECIPE ---")
    save = input("Do you want to save this recipe to favorites? (yes/no): ").strip().lower()
    notes = input("Any personal notes you'd like to save with it? (optional): ").strip()

    favorites = state.get("favorites", []) or []
    if save == "yes":
        favorites.append(recipe)
        print("Recipe saved to favorites!")

    return {
        "favorites": favorites,
        "user_notes": notes if notes else ""
    }


def route_after_generation(state: RecipeState) -> str:
    """Determine flow after recipe generation"""
    recipe = state.get("generated_recipe", "")
    if "Error" in recipe:
        print("\nSkipping remaining nodes due to generation error")
        return "FeedbackNode"
        
    dietary_restrictions = state.get("dietary_restrictions", [])
    if dietary_restrictions:
        return "DietAdjustmentNode"
    else:
        return "IngredientSubstitutionNode"


# Main function to run the application
def main():
    # Build the graph
    graph = StateGraph(RecipeState)

    # Add nodes
    graph.add_node("UserInputNode", user_input_node)
    graph.add_node("RecipeGenerationNode", recipe_generation_node)
    graph.add_node("DietAdjustmentNode", diet_adjustment_node)
    graph.add_node("IngredientSubstitutionNode", ingredient_substitution_node)
    graph.add_node("FeedbackNode", feedback_node)
    graph.add_node("StorageNode", storage_node)

    # Set entry point
    graph.set_entry_point("UserInputNode")

    # Add conditional edges
    graph.add_conditional_edges(
        "RecipeGenerationNode",
        route_after_generation,
        {
            "DietAdjustmentNode": "DietAdjustmentNode",
            "IngredientSubstitutionNode": "IngredientSubstitutionNode",
            "FeedbackNode": "FeedbackNode"
        }
    )

    # Add regular edges
    graph.add_edge("UserInputNode", "RecipeGenerationNode")
    graph.add_edge("DietAdjustmentNode", "IngredientSubstitutionNode")
    graph.add_edge("IngredientSubstitutionNode", "FeedbackNode")
    graph.add_edge("FeedbackNode", "StorageNode")
    graph.add_edge("StorageNode", END)

    # Compile the graph
    app = graph.compile()

    # Initialize with empty state
    initial_state = {
        "ingredients": [],
        "dietary_restrictions": [],
        "preferences": [],
        "generated_recipe": None,
        "adjusted_recipe": None,
        "substitutions": None,
        "meal_plan": None,
        "shopping_list": None,
        "feedback": None,
        "favorites": [],
        "user_notes": None
    }

    print("\n==== Recipe Assistant ====")
    print("This app helps you create recipes based on available ingredients and preferences.")
    
    try:
        # Run the graph with the initial state
        final_state = app.invoke(initial_state)

        print("\n==== Final State ====")
        for key, value in final_state.items():
            if value:  # Only print non-empty values
                print(f"\n{key.upper()}:")
                print(value)
    except Exception as e:
        print(f"\nError running the application: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()