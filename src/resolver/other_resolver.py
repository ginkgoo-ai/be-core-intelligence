import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.enum import EnumOutputParser

from langchain_unstructured import UnstructuredLoader
from fastapi import HTTPException
from src.document_struction import *
from langchain_core.prompts import PromptTemplate
from fastapi.responses import JSONResponse
import asyncio
from src.model.file_definition import ClassificationCategory,ConsolidatedDocumentModelPartOne, ConsolidatedDocumentModelPartTwo


class OtherDocumentAnalysis:

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

    def _build_prompt(self) -> str:
        """
        Builds the prompt for the Gemini model based on classification guidelines
        and document information.
        """
        # Classification guidelines provided by the user
        prompt_template = """
        Extract the information about the role from the following text.
        Please output strictly in the following JSON format:
        {format_instructions}
        
        Text content
        {text_input}
        """

        return prompt_template

    async def analyze_document_by_path(self, doc_url: str) :
        try:

            # 加载文档
            try:
                loader = UnstructuredLoader(web_url=doc_url)
                documents = loader.load()

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"无法加载文件: {str(e)}")


            context = ""
            for item in documents:
                context += item.page_content

            prompt_template_text = self._build_prompt()

            parser_part_one = PydanticOutputParser(pydantic_object=ConsolidatedDocumentModelPartOne)
            prompt_one = PromptTemplate(  # 使用不同的变量名以提高清晰度
                template=prompt_template_text,
                input_variables=["text_input"],
                partial_variables={"format_instructions": parser_part_one.get_format_instructions()}
            )
            chain_one = prompt_one | self.model | parser_part_one

            parser_part_two = PydanticOutputParser(pydantic_object=ConsolidatedDocumentModelPartTwo)
            prompt_two = PromptTemplate(  # 使用不同的变量名
                template=prompt_template_text,  # 相同的模板文本
                input_variables=["text_input"],
                partial_variables={"format_instructions": parser_part_two.get_format_instructions()}
            )
            chain_two = prompt_two | self.model | parser_part_two

            # 创建两个任务
            task_one = chain_one.ainvoke({"text_input": context})
            task_two = chain_two.ainvoke({"text_input": context})


            response_part_one: ConsolidatedDocumentModelPartOne
            response_part_two: ConsolidatedDocumentModelPartTwo
            response_part_one, response_part_two = await asyncio.gather(
                task_one,
                task_two
            )


            merged_dict = {**response_part_one.model_dump(), **response_part_two.model_dump()}

            return merged_dict

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"处理文件时出错: {str(e)}"}
            )

    async def classify_document(self, doc_url: str, prompt_template_text: str) -> ClassificationCategory:
        try:

            # 加载文档
            try:
                loader = UnstructuredLoader(web_url=doc_url)
                documents = loader.load()

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"error load files: {str(e)}")


            context = ""
            for item in documents:
                context += item.page_content

            classification_category = EnumOutputParser(enum=ClassificationCategory)
            prompt = PromptTemplate(  # 使用不同的变量名以提高清晰度
                template=prompt_template_text,
                input_variables=["text_input"],
                partial_variables={"format_instructions": classification_category.get_format_instructions()}
            )
            chain = prompt | self.model | classification_category

            response = chain.invoke({"text_input": context})

            return ClassificationCategory(response)

        except Exception as e:
            return ClassificationCategory.UNKNOWN