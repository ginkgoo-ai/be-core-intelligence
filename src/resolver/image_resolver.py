import asyncio

from langchain.output_parsers.enum import EnumOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from src.document_struction import *
from src.model.file_definition import ClassificationCategory,ConsolidatedDocumentModelPartOne, ConsolidatedDocumentModelPartTwo


class ImageDocumentAnalysis:

    def __init__(self):
        """
        Initializes the Document.
        In a real scenario, this would initialize the Gemini API client.

        Args:
            api_key: The API key for the Gemini service. (Currently unused in simulation)
        """
        self.api_key = os.getenv("MODEL_API_KEY")
        self.model = ChatOpenAI(
            base_url=os.getenv("MODEL_BASE_URL"),
            api_key=self.api_key,
            model=os.getenv("MODEL_NAME"),
            temperature=0,
            max_tokens=2048  # Increased max_tokens for potentially larger content
        )

    def get_content(self, doc_url: str, format_json: str):
        return [
            {
                "type": "text",
                "text": format_json,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": doc_url
                }
            }
        ]

    async def analyze_document_by_path(self, doc_url: str) -> dict:


        # --- Part One ---
        parser_part_one = PydanticOutputParser(pydantic_object=ConsolidatedDocumentModelPartOne)


        chain_one = self.model | parser_part_one

        # --- Part Two ---
        parser_part_two = PydanticOutputParser(pydantic_object=ConsolidatedDocumentModelPartTwo)


        chain_two = self.model | parser_part_two


        task_one = chain_one.ainvoke([HumanMessage(content=self.get_content(doc_url, parser_part_one.get_format_instructions()))])
        task_two = chain_two.ainvoke([HumanMessage(content=self.get_content(doc_url, parser_part_two.get_format_instructions()))])


        response_part_one: ConsolidatedDocumentModelPartOne
        response_part_two: ConsolidatedDocumentModelPartTwo
        response_part_one, response_part_two = await asyncio.gather(
            task_one,
            task_two
        )


        merged_dict = {**response_part_one.model_dump(), **response_part_two.model_dump()}

        return merged_dict

    async def classify_document(self, doc_url: str, prompt_template_text: str) -> ClassificationCategory:

        message_content_parts = [
            {
                "type": "text",
                "text": prompt_template_text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": doc_url
                }
            }
        ]

        classification_category = EnumOutputParser(enum=ClassificationCategory)

        chain = self.model | classification_category

        message = HumanMessage(content=message_content_parts)

        response = chain.invoke([message])

        return ClassificationCategory(response)

