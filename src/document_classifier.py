import os

from src.model import file_definition
from src.resolver import image_resolver, other_resolver, pdf_resolver
from urllib.parse import urlparse

# --- Entities ---



# --- Classifier ---

class DocumentClassifier:
    """
    A classifier that uses a (simulated) Gemini model to categorize documents
    for UK Skilled Worker Visa applications based on provided guidelines.
    """
    def __init__(self, doc_url: str):
        """
        Initializes the classifier.

        Args:
            doc_url: document url
        """
        self._doc_url = doc_url


    def _build_prompt(self) -> str:
        """
        Builds the prompt for the Gemini model based on classification guidelines
        and document information.
        """
        # Classification guidelines provided by the user
        prompt_template = """
                You are an expert in UK Skilled Worker Visa document classification.
                Your task is to classify the provided document based on the guidelines below.
                Please return ONLY one of the following category names:
                - "Identification Documents"
                - "Mandatory Documents"
                - "Optional Documents"
                
                If the document does not clearly fit into any of these categories based on the provided information, or if the information is insufficient, return "Unknown".
                
                ## Classification Guidelines - SKILLED WORKER VISA SPECIFIC
                
                ### 1. Identification Documents
                Personal identification materials required for all applicants.
                Examples:
                - Current passport
                - Proof of current home address (utility bills, bank statements, etc.)
                - National identification card
                
                ### 2. Mandatory Documents
                Essential documents required for all Skilled Worker Visa applications.
                Examples:
                - Certificate of Sponsorship (CoS) from UK employer
                - Job Offer letter from UK employer
                - CV/Resume
                - Evidence of English language proficiency (test results, degree certificates, etc.)
                
                ### 3. Optional Documents
                Supplementary documents that may be required based on specific circumstances or that provide additional supporting evidence.
                Examples:
                - Bank statements showing sufficient funds to live in the UK
                - Tuberculosis (TB) test certificate (for nationals from countries listed in Appendix T)
                - Criminal record certificate (required for specific SOC codes/professions)
                - Any other supporting documents that cannot be classified in the previous categories but might strengthen the application

                
            """

        return prompt_template

    def classify_document(self):

        parsed_url = urlparse(self._doc_url)
        _, file_extension = os.path.splitext(parsed_url.path)

        file_extension = file_extension[1:].lower()
        if not file_extension:
            return other_resolver.OtherDocumentAnalysis().classify_document(self._doc_url, self._build_prompt())

        if file_extension in file_definition.PDF_EXTENSIONS:
            return pdf_resolver.GeminiPDFDocumentAnalysis.classify_document(self._doc_url, self._build_prompt())

        elif file_extension in file_definition.IMAGE_EXTENSIONS:
            return image_resolver.ImageDocumentAnalysis.classify_document(self._doc_url, self._build_prompt())

        else:
            return other_resolver.OtherDocumentAnalysis().classify_document(self._doc_url, self._build_prompt())



