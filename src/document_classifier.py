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

        prompt_template = """
                You are an expert in UK Skilled Worker Visa document classification.
                Your task is to classify the provided document based on the guidelines below.
                Please return ONLY one of the following category names:
                    - PASSPORT
                        Description: passport file
                    - PROOF_OF_CURRENT_HOME_ADDRESS 
                        Description: Proof of current home address, have home address information
                    - NATIONAL_IDENTIFICATION_CARD 
                        Description: National identification card
                    - COPY_OF_CERTIFICATE_OF_SPONSORSHIP 
                        Description: Copy of Certificate of Sponsorship (CoS)
                    - JOB_OFFER_LETTER_FROM_NEW_EMPLOYER
                        Description: Job Offer letter from new employer
                    - UP_TO_DATE_CV 
                        Description: Up to date CV
                    - EVIDENCE_OF_SATISFYING_THE_ENGLISH_LANGUAGE_REQUIREMENT
                        Description: Evidence of satisfying the English language requirement
                    - BANK_STATEMENTS_SHOWING_SUFFICIENT_FUNDS_TO_LIVE_IN_THE_UK 
                        Description: Bank statements showing sufficient funds to live in the UK
                    - TB_TEST_CERTIFICATE
                        Description: TB test certificate (Tuberculosis test certificate)
                    - CRIMINAL_RECORD_CERTIFICATE
                        Description: Criminal record certificate

                If the document does not clearly fit into any of these categories based on the provided information, or if the information is insufficient, return "Unknown".

        """

        return prompt_template

    def classify_document(self):

        parsed_url = urlparse(self._doc_url)
        _, file_extension = os.path.splitext(parsed_url.path)

        file_extension = file_extension[1:].lower()
        if not file_extension:
            return other_resolver.OtherDocumentAnalysis().classify_document(self._doc_url, self._build_prompt())

        if file_extension in file_definition.PDF_EXTENSIONS:
            return pdf_resolver.GeminiPDFDocumentAnalysis().classify_document(self._doc_url, self._build_prompt())

        elif file_extension in file_definition.IMAGE_EXTENSIONS:
            return image_resolver.ImageDocumentAnalysis().classify_document(self._doc_url, self._build_prompt())

        else:
            return other_resolver.OtherDocumentAnalysis().classify_document(self._doc_url, self._build_prompt())



