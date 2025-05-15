import os

from src.model.file_definition import ClassificationCategory,ConsolidatedDocumentModelPartOne, ConsolidatedDocumentModelPartTwo
from google import genai
from google.genai import types
import httpx
import asyncio
class GeminiPDFDocumentAnalysis:

    def __init__(self):
        """
        Initializes the Document.
        In a real scenario, this would initialize the Gemini API client.

        Args:
            api_key: The API key for the Gemini service. (Currently unused in simulation)
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODE")
        self.client = genai.Client(api_key=self.api_key)
        self.model = self.client.models

    def _build_prompt(self) -> str:
        """
        Builds the prompt for the Gemini model based on classification guidelines
        and document information.
        """
        # Classification guidelines provided by the user
        prompt_template = """
        Analyze the following document content and structure the extracted information 
        strictly according to the JSON schema provided below. The output MUST be a valid JSON object 
        that conforms to this schema. Populate all fields in the schema with relevant information 
        found in the document. If information for a field is not found, use the default value specified 
        in the schema (e.g., null).

        """

        return prompt_template

    async def analyze_document_by_path(self, doc_url: str):
        """
        Structure a given document using the (simulated) Gemini model.

        Args:
            doc_url: Document url

        Returns:
            The structure result.
        """

        doc_data = httpx.get(doc_url).content

        prompt = self._build_prompt()

        if self.model:
            print("Using actual Gemini API call...")
            try:
                tasks = [self.client.aio.models.generate_content(model=self.model_name,
                                                     contents=[
                                                         types.Part.from_bytes(
                                                             data=doc_data,
                                                             mime_type='application/pdf',
                                                         ),
                                                         prompt],
                                                     config={
                                                         'response_mime_type': 'application/json',
                                                         'response_schema': ConsolidatedDocumentModelPartOne,
                                                     }),
                         self.client.aio.models.generate_content(model=self.model_name,
                                                     contents=[
                                                         types.Part.from_bytes(
                                                             data=doc_data,
                                                             mime_type='application/pdf',
                                                         ),
                                                         prompt],
                                                     config={
                                                         'response_mime_type': 'application/json',
                                                         'response_schema': ConsolidatedDocumentModelPartTwo,
                                                     })]

                results_list = await asyncio.gather(*tasks)
                flattened_results_dict = {}
                for i, result_text in enumerate(results_list):
                    flattened_results_dict = {**flattened_results_dict, **result_text.parsed.__dict__}

                return flattened_results_dict

            except Exception as e:
                print(f"Error calling Gemini API: {e}")
    async def classify_document(self, doc_url: str, prompt_template_text: str) -> ClassificationCategory:
        """
        Classifies a given document using the (simulated) Gemini model.

        Args:
            doc_url: Document url.
            prompt_template_text: prompt template text.
        Returns:
            The classification result.
        """
        category_enum = ClassificationCategory.UNKNOWN
        doc_data = httpx.get(doc_url).content
        if self.model:
            print("Using actual Gemini API call...")
            actual_gemini_response_text = "Unknown"
            try:
                # Actual Gemini API call
                response = self.model.generate_content(model=self.model_name,
                                                       contents=[
                                                           types.Part.from_bytes(
                                                               data=doc_data,
                                                               mime_type='application/pdf',
                                                           ),
                                                           prompt_template_text],
                                                       config={
                                                           'response_mime_type': 'text/x.enum',
                                                           'response_schema': ClassificationCategory,
                                                       })
                if response and response.text is not None:
                    actual_gemini_response_text = response.text.strip()
                    category_enum = ClassificationCategory(actual_gemini_response_text)
                else:
                    print("Warning: Gemini API response text was None in classify_document.")

                return category_enum
            except Exception as e:
                print(
                    f"Warning: Gemini response {actual_gemini_response_text} is not a defined category. Defaulting to UNKNOWN. e:{e}")

        return category_enum