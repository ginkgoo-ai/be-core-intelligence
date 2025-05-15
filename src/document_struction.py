
from resolver import image_resolver, other_resolver, pdf_resolver
from src.model import file_definition
from urllib.parse import urlparse
import os


class DocumentResolver:
    def __init__(self, doc_url: str):
        """
        Initializes the resolver.

        Args:
            doc_url: the doc url
        """
        self._doc_url = doc_url

    def resolver(self):
        parsed_url = urlparse(self._doc_url)
        _, file_extension = os.path.splitext(parsed_url.path)

        file_extension = file_extension[1:].lower()

        if not file_extension:
            return other_resolver.OtherDocumentAnalysis().analyze_document_by_path(self._doc_url)

        if file_extension in file_definition.PDF_EXTENSIONS:

            return pdf_resolver.GeminiPDFDocumentAnalysis().analyze_document_by_path(self._doc_url)

        elif file_extension in file_definition.IMAGE_EXTENSIONS:
            return image_resolver.ImageDocumentAnalysis().analyze_document_by_path(self._doc_url)

        else:
            return other_resolver.OtherDocumentAnalysis().analyze_document_by_path(self._doc_url)
