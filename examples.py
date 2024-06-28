from rich.console import Console

from llm_reflection import GoogleParams
from llm_reflection import LlmReflection
from llm_reflection import PrompTemplate

console = Console()


def example_translation():
    source_lang = "english"
    target_lang = "spanish"
    source_text = """
        OpenAI is American artificial intelligence (AI) research laboratory consisting of
        the non-profit OpenAI Incorporated and its for-profit subsidiary corporation OpenAI
        Limited Partnership. OpenAI conducts AI research with the declared intention of
        promoting and developing a friendly AI. OpenAI systems run on an Azure-based supercomputing
        platform from Microsoft.
        The OpenAI API is powered by a diverse set of models with different capabilities and price points.
    """

    prompt = PrompTemplate(
        person="You are an expert linguist, specializing in translation",
        task=f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text.
        Do not provide any explanations or text apart from the translation.
        {source_lang}:{source_text}
        {target_lang}:""",
        output_format="Do not provide any explanations or text apart from the translation.",
    )

    llm_reflection = LlmReflection(
        GoogleParams(model_name="gemini-1.5-flash", temperature=0.8),
        system_message="You are an expert linguist, specializing in translation",
    )

    translation = llm_reflection.generate_text(
        prompt,
        reflection_items=[
            "accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text)",
            "fluency(by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions)",
            "style(by ensuring the translations reflect the style of the source text and takes into account any cultural context)",
            "terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).",
        ],
    )

    console.print("*" * 10)
    console.print(translation)


def example_recipe():
    country = "Mexico"
    ingredients = ["rice", "meat", "vegetables"]

    prompt = PrompTemplate(
        person=f"You are an expert cooking and the best chef. Create recipes with these food ingredients.\
                    You are from {country}",
        task=f"""Create 1 recipe with these food ingredients: {ingredients}""",
        output_format="JSON",
    )

    llm_reflection = LlmReflection(
        GoogleParams(model_name="gemini-1.5-flash", temperature=0.8),
        system_message=f"You are an expert cooking and the best chef. Create recipes with these food ingredients.\
                    You are from {country}",
    )

    recipe = llm_reflection.generate_text(
        prompt,
        reflection_items=[
            "feasibility of the recipe (availability of ingredients, cooking techniques).",
            "feasibility of the recipe (availability of ingredients, cooking techniques).",
            "cooking techniques (Are there specific ways dishes are prepared in this country?)",
            "flavors (Is the cuisine known for being spicy, savory, sweet, or something else entirely) ",
            "presentation (How is food typically presented in {country})",
        ],
    )

    console.print("*" * 30)
    console.print("History")
    console.print(llm_reflection.history)
    console.print("*" * 30)
    console.print(recipe)


def example_recipe_without_reflection_point():
    country = "Mexico"
    ingredients = ["rice", "meat", "vegetables"]

    prompt = PrompTemplate(
        person=f"You are an expert cooking and the best chef. Create recipes with these food ingredients.\
                    You are from {country}",
        task=f"""Create 1 recipe with these food ingredients: {ingredients}""",
        output_format="JSON",
    )

    llm_reflection = LlmReflection(
        GoogleParams(model_name="gemini-1.5-flash", temperature=0.8),
        system_message=f"You are an expert cooking and the best chef. Create recipes with these food ingredients.\
                    You are from {country}",
    )

    recipe = llm_reflection.generate_text(prompt)

    console.print("*" * 30)
    console.print("History")
    console.print(llm_reflection.history)
    console.print("*" * 30)
    console.print(recipe)


if __name__ == "__main__":
    example_recipe_without_reflection_point()
