from dataclasses import dataclass
from typing import List

from langchain_google_vertexai import VertexAI
from langchain_openai import OpenAI
from rich.console import Console

console = Console()


@dataclass
class ProviderParams:
    """
    Base class for provider parameters.

    Attributes:
        model_name (str): The name of the model to use.
        temperature (float): The temperature parameter for the model.
    """

    model_name: str
    temperature: float


@dataclass
class OpenAIParams(ProviderParams):
    """
    Parameters for OpenAI models.

    Attributes:
        model_name (str): The name of the model to use.
        temperature (float): The temperature parameter for the model.
        openai_api_key (str): Your OpenAI API key.
        openai_organization (str): Your OpenAI organization ID.
    """

    openai_api_key: str
    openai_organization: str


@dataclass
class GoogleParams(ProviderParams):
    pass


@dataclass
class PrompTemplate:
    """
    A class to represent a prompt template.

    Attributes:
        person (str): The persona of the LLM.
        task (str): The task to be performed by the LLM.
        context (str): Additional context for the LLM.
        output_format (str): The desired output format for the LLM.
    """

    person: str
    task: str
    context: str = ""
    output_format: str = ""

    @property
    def prompt_text(self):
        return self.person + self.task + self.context + self.output_format


class LlmReflection:
    """
    A class to represent a LLM reflection object.

    Attributes:
        providerParams (ProviderParams): The parameters for the LLM provider.
        system_message (str): The system message for the LLM.
    """

    def __init__(self, providerParams: ProviderParams, system_message: str):
        self.system_message = system_message
        self.llm = self._create_llm(providerParams, system_message)
        self._history: List[str] = []

    @property
    def history(self) -> List[str]:
        """
        Returns the history of LLM interactions.

        Returns:
            List[str]: The history of LLM interactions.
        """
        return self._history

    def generate_text(self, prompt: PrompTemplate, reflection_items: List = []) -> str:
        """
        Generates text using the LLM with reflection.

        Args:
            prompt (PrompTemplate): The prompt template to use.
            reflection_items (List, optional): A list of reflection items to consider. Defaults to [].

        Returns:
            str: The generated text.
        """
        self._history = []

        reflection_items_prompt = ""
        for index, item in enumerate(reflection_items):
            reflection_items_prompt += f"""
                {index + 1}. {item} \n
            """

        first_text = self._generate_first_text(prompt)
        reflection = self._generate_reflection(
            prompt, first_text, reflection_items_prompt
        )
        return self._generate_improve_text(
            prompt, first_text, reflection, reflection_items_prompt
        )

    def _generate_first_text(self, prompt: PrompTemplate) -> str:
        result = self._llm_invoke(prompt.prompt_text)
        self._history.append(result)
        return result

    def _generate_reflection(
        self,
        prompt: PrompTemplate,
        first_text: str,
        reflection_items_prompt: str,
    ) -> str:
        """
        Generates reflection suggestions from the LLM.

        Args:
            prompt (PrompTemplate): The prompt template to use.
            first_text (str): The first generated text.
            reflection_items_prompt (str): The prompt for reflection items.

        Returns:
            str: The reflection suggestions.
        """

        if reflection_items_prompt != "":
            prompt_reflection = f"""
                you are {prompt.person} and then give constructive criticism and helpful suggestions to improve the following task
                {prompt.task}

                The first result delimited by XML tags <FIRST_RESULT></FIRST_RESULT> is the follow:
                <FIRST_RESULT>
                {first_text}
                </FIRST_RESULT>

                When writing suggestions, pay attention to whether there are ways to improve \n\
                {reflection_items_prompt}

                Write a list of specific, helpful and constructive suggestions for improving the {prompt.task}.
                Output only the suggestions and nothing else.
            """
        else:
            prompt_reflection = f"""
                you are {prompt.person} and then give constructive criticism and helpful suggestions to improve the following task
                {prompt.task}

                The first result delimited by XML tags <FIRST_RESULT></FIRST_RESULT> is the follow:
                <FIRST_RESULT>
                {first_text}
                </FIRST_RESULT>

                When writing suggestions, first define four reflection points for this task and pay attention to whether there are ways to improve \n\

                Write a list of specific, helpful and constructive suggestions for improving the {prompt.task}.
                Output only the suggestions and nothing else.
            """

        result = self._llm_invoke(prompt_reflection)
        self._history.append(result)
        return result

    def _generate_improve_text(
        self,
        prompt: PrompTemplate,
        first_text: str,
        expert_suggestions: str,
        reflection_items_prompt: str,
    ) -> str:
        """
        Generates improved text based on reflection suggestions.

        Args:
            prompt (PrompTemplate): The prompt template to use.
            first_text (str): The first generated text.
            expert_suggestions (str): The reflection suggestions.
            reflection_items_prompt (str): The prompt for reflection items.

        Returns:
            str: The improved text.
        """

        prompt_improve_text = f"""
            Your task is to carefully read, then edit, a {prompt.task}, taking into
            account a list of expert suggestions and constructive criticisms.

            The first result delimited by XML tags <FIRST_RESULT></FIRST_RESULT> is the follow:
            <FIRST_RESULT>
            {first_text}
            </FIRST_RESULT>,

            and the expert suggestions are delimited by XML tags <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
            as follows:

            <EXPERT_SUGGESTIONS>
            {expert_suggestions}
            </EXPERT_SUGGESTIONS>

            Please take into account the expert suggestions when you are doing {prompt.task}. Edit the first result by ensuring:
            {reflection_items_prompt}

            Output only the new result and nothing else
        """

        result = self._llm_invoke(prompt_improve_text)
        self._history.append(result)
        return result

    def _llm_invoke(
        self,
        prompt: str,
    ) -> str:
        return self.llm.invoke(prompt)

    def _create_llm(
        self,
        providerParams: ProviderParams,
        system_message: str,
    ) -> VertexAI | OpenAI:
        """
        Creates the LLM object based on the provider parameters.

        Args:
            providerParams (ProviderParams): The parameters for the LLM provider.
            system_message (str): The system message for the LLM.

        Returns:
            VertexAI | OpenAI: The LLM object.
        """

        if isinstance(providerParams, OpenAIParams):
            return OpenAI(
                openai_api_key=providerParams.openai_api_key,
                openai_organization=providerParams.openai_organization,
                model_name=providerParams.model_name,
                system_message=system_message,
            )

        return VertexAI(
            model_name=providerParams.model_name, system_message=system_message
        )
