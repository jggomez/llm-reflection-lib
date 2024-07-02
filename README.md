# LLM Reflection Library

This library provides a simple way to implement reflection in your LLM applications. Reflection is the process of having an LLM evaluate its own output and provide suggestions for improvement. This can be a powerful technique for improving the quality of LLM outputs.

This Library is based on [Translation Agent: Agentic translation using reflection workflow repository](https://github.com/andrewyng/translation-agent) and this article [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)

## Installation

```bash
pip install llm-reflection
```

## Usage
The library provides a LlmReflection class that you can use to interact with an LLM and implement reflection.

Here is an example of how to use the library to translate a text from English to Spanish:

```python
from llm_reflection import LlmReflection
from llm_reflection import GoogleParams
from llm_reflection import PrompTemplate
from rich.console import Console

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
        output_format="Do not provide any explanations or text apart from the translation."
    )

    llm_reflection = LlmReflection(
        GoogleParams(
            model_name="gemini-1.5-flash",
            temperature=0.8
        ),
        system_message="You are an expert linguist, specializing in translation"
    )

    translation = llm_reflection.generate_text(
        prompt,
        reflection_items=[
            "accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text)",
            "fluency(by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions)",
            "style(by ensuring the translations reflect the style of the source text and takes into account any cultural context)",
            "terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).",
        ]
    )

    console.print("*"*10)
    console.print(translation)

```

## This code will:

* **Create a prompt template:** This defines the persona of the Large Language Model (LLM), the task to be performed, and any additional context.
* **Create a LLMReflection object:** This object will be used to interact with the LLM.
* **Generate text:** The `generate_text` method will generate text using the LLM and then provide reflection suggestions.

## Reflection Items (These may or may not be used)

The `reflection_items` parameter in the `generate_text` method is a list of reflection points that the LLM should consider when evaluating its output. These reflection points should be specific and actionable.

For example, in the translation example above, the reflection points are:

* **Accuracy:** The LLM should check for errors in the translation, such as mistranslations, omissions, or additions.
* **Fluency:** The LLM should ensure that the translation is grammatically correct and flows naturally.
* **Style:** The LLM should ensure that the translation reflects the style of the source text.
* **Terminology:** The LLM should ensure that the translation uses appropriate terminology and idioms.

## Supported LLMs

The library currently supports the following LLMs:

* **Google Vertex AI:** You can use the `GoogleParams` class to configure the LLM.
* **OpenAI:** You can use the `OpenAIParams` class to configure the LLM.

Made with ❤ by  [jggomez](https://devhack.co).

[![Twitter Badge](https://img.shields.io/badge/-@jggomezt-1ca0f1?style=flat-square&labelColor=1ca0f1&logo=twitter&logoColor=white&link=https://twitter.com/jggomezt)](https://twitter.com/jggomezt)
[![Linkedin Badge](https://img.shields.io/badge/-jggomezt-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/jggomezt/)](https://www.linkedin.com/in/jggomezt/)
[![Medium Badge](https://img.shields.io/badge/-@jggomezt-03a57a?style=flat-square&labelColor=000000&logo=Medium&link=https://medium.com/@jggomezt)](https://medium.com/@jggomezt)

## License

    Copyright 2024 Juan Guillermo Gómez

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
