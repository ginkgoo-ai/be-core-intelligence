import enum
from pydantic import BaseModel
from typing import Optional, List, Any, Dict

from dataclasses import dataclass
from dataclasses_json import dataclass_json,Undefined

PDF_EXTENSIONS = {'pdf'}

IMAGE_EXTENSIONS = {
    'jpg', 'jpeg', 'jpe', 'jif', 'jfif', 'jfi',  # JPEG 文件
    'png',                                      # Portable Network Graphics
    'gif',                                      # Graphics Interchange Format
    'bmp', 'dib',                               # Windows Bitmap
    'tiff', 'tif',                              # Tagged Image File Format
    'webp',                                     # WebP
    'heic', 'heif',                             # High Efficiency Image File Format (Apple)
    'ico',                                      # Icon file
    'avif',                                     # AV1 Image File Format
    'apng',                                     # Animated Portable Network Graphics
    'svg',                                      # Scalable Vector Graphics
}


class ClassificationCategory(enum.Enum):
    """Enumeration for document classification categories."""
    PASSPORT = "passport"
    PROOF_OF_CURRENT_HOME_ADDRESS = "Proof of current home address"
    NATIONAL_IDENTIFICATION_CARD = "National identification card"
    COPY_OF_CERTIFICATE_OF_SPONSORSHIP = "Copy of Certificate of Sponsorship (CoS)"
    JOB_OFFER_LETTER_FROM_NEW_EMPLOYER = "Job Offer letter from new employer"
    UP_TO_DATE_CV = "Up to date CV"
    EVIDENCE_OF_SATISFYING_THE_ENGLISH_LANGUAGE_REQUIREMENT = "Evidence of satisfying the English language requirement"
    BANK_STATEMENTS_SHOWING_SUFFICIENT_FUNDS_TO_LIVE_IN_THE_UK = "Bank statements showing sufficient funds to live in the UK"
    TB_TEST_CERTIFICATE = "TB test certificate (Tuberculosis test certificate)"
    CRIMINAL_RECORD_CERTIFICATE = "Criminal record certificate"
    UNKNOWN = "Unknown" # For cases where classification fails or is unclear


# Utility bill.json
@dataclass_json
@dataclass
class UtilityBillBankAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    postcode: Optional[str] = None

@dataclass_json
@dataclass
class UtilityBillBank(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    address: Optional[UtilityBillBankAddress] = None

@dataclass_json
@dataclass
class UtilityBillTransactionDetails(BaseModel):
    reference_number: Optional[str] = None
    credit_account_number: Optional[str] = None
    amount: Optional[str] = None # Could be Decimal or float
    standard_fee_payable: Optional[str] = None # Could be Decimal or float

@dataclass_json
@dataclass
class UtilityBillPaymentInstructionsBankDetails(BaseModel):
    bank_name: Optional[str] = None
    account_name: Optional[str] = None
    sort_code: Optional[str] = None
    account_number: Optional[str] = None
    iban: Optional[str] = None
    swift_bic: Optional[str] = None

@dataclass_json
@dataclass
class UtilityBillPaymentInstructions(BaseModel):
    cheque_acceptance: Optional[str] = None
    bank_details: Optional[UtilityBillPaymentInstructionsBankDetails] = None
    cash_payment_note: Optional[str] = None

@dataclass_json
@dataclass
class UtilityBillPayeeAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    postcode: Optional[str] = None

@dataclass_json
@dataclass
class UtilityBillPayee(BaseModel):
    name: Optional[str] = None
    address: Optional[UtilityBillPayeeAddress] = None

@dataclass_json
@dataclass
class UtilityBillAdditionalCodes(BaseModel):
    code_1: Optional[str] = None
    code_2: Optional[str] = None
    code_3: Optional[str] = None

@dataclass_json
@dataclass
class UtilityBillNotes(BaseModel):
    do_not_write_below_line: Optional[str] = None
    do_not_fold_counterfoil: Optional[str] = None

@dataclass_json(undefined=Undefined.EXCLUDE) # 配置 dataclass_json 忽略未定义字段
@dataclass
class UtilityBillModel(BaseModel):
    bank: Optional[UtilityBillBank] = None
    transaction_details: Optional[UtilityBillTransactionDetails] = None
    payment_instructions: Optional[UtilityBillPaymentInstructions] = None
    payee: Optional[UtilityBillPayee] = None
    signature: Optional[str] = None
    date: Optional[str] = None # Could be date
    additional_codes: Optional[UtilityBillAdditionalCodes] = None
    notes: Optional[UtilityBillNotes] = None

# Referee info.json
@dataclass_json
@dataclass
class RefereeInfoAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    postcode: Optional[str] = None

@dataclass_json
@dataclass
class RefereeInfoDetails(BaseModel):
    family_name: Optional[str] = None
    given_name: Optional[str] = None
    date_of_birth: Optional[str] = None # Could be date
    address: Optional[RefereeInfoAddress] = None
    previous_address: Optional[str] = None
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    profession: Optional[str] = None
    british_passport: Optional[str] = None # Could be bool
    passport_number: Optional[str] = None
    relationship_to_applicant: Optional[str] = None

@dataclass_json
@dataclass
class RefereeInfoModel(BaseModel):
    referee_details: Optional[RefereeInfoDetails] = None

# Referee and identity.json
@dataclass_json
@dataclass
class ApplicantIdentity(BaseModel):
    name: Optional[str] = None

@dataclass_json
@dataclass
class RefereeIdentity(BaseModel):
    full_name: Optional[str] = None
    signature: Optional[str] = None
    date: Optional[str] = None # Could be date

@dataclass_json
@dataclass
class RefereeAndIdentityModel(BaseModel):
    applicant: Optional[ApplicantIdentity] = None
    referee_1: Optional[RefereeIdentity] = None
    referee_2: Optional[RefereeIdentity] = None

# passport.json
@dataclass_json
@dataclass
class PassportMachineReadableZone(BaseModel):
    document_type: Optional[str] = None
    country_code: Optional[str] = None
    passport_number: Optional[str] = None
    check_digit: Optional[str] = None
    date_of_birth: Optional[str] = None # Could be date
    sex: Optional[str] = None
    expiry_date: Optional[str] = None # Could be date
    nationality: Optional[str] = None
    check_digit_2: Optional[str] = None
    check_digit_3: Optional[str] = None

@dataclass_json
@dataclass
class PassportStamp(BaseModel):
    location: Optional[str] = None
    date: Optional[str] = None # Could be date
    officer_id: Optional[str] = None
    notes: Optional[str] = None

@dataclass_json
@dataclass
class PassportModel(BaseModel):
    passport_number: Optional[str] = None
    surname: Optional[str] = None
    given_name: Optional[str] = None
    nationality: Optional[str] = None
    date_of_birth: Optional[str] = None # Could be date
    place_of_issue: Optional[str] = None
    date_of_issue: Optional[str] = None # Could be date
    date_of_expiry: Optional[str] = None # Could be date
    type: Optional[str] = None
    number_of_entries: Optional[str] = None
    valid_from: Optional[str] = None # Could be date
    valid_until: Optional[str] = None # Could be date
    vaf_number: Optional[str] = None
    observations: Optional[str] = None
    conditions: Optional[str] = None
    machine_readable_zone: Optional[PassportMachineReadableZone] = None
    stamps: Optional[List[PassportStamp]] = None

# Parents info.json
@dataclass_json
@dataclass
class ParentDetail(BaseModel):
    relationship_to_applicant: Optional[str] = None
    given_names: Optional[str] = None
    family_name: Optional[str] = None
    date_of_birth: Optional[str] = None # Could be date
    town_of_birth: Optional[str] = None
    country_of_birth: Optional[str] = None
    country_of_nationality: Optional[str] = None
    always_same_nationality: Optional[str] = None # Could be bool
    nationality_at_applicant_birth: Optional[str] = None

@dataclass_json
@dataclass
class ParentsInfoModel(BaseModel):
    parent_one: Optional[ParentDetail] = None
    parent_two: Optional[ParentDetail] = None

# P60.json
@dataclass_json
@dataclass
class P60EmployeeInstructions(BaseModel):
    purpose: Optional[str] = None
    legal_requirement: Optional[str] = None

@dataclass_json
@dataclass
class P60EmployeeDetails(BaseModel):
    surname: Optional[str] = None
    forenames_or_initials: Optional[str] = None
    national_insurance_number: Optional[str] = None

@dataclass_json
@dataclass
class P60Pay(BaseModel):
    previous_employment: Optional[str] = None # Could be Decimal or float
    current_employment: Optional[str] = None # Could be Decimal or float
    total_for_year: Optional[str] = None # Could be Decimal or float

@dataclass_json
@dataclass
class P60IncomeTax(BaseModel):
    previous_employment: Optional[str] = None # Could be Decimal or float
    current_employment: Optional[str] = None # Could be Decimal or float
    total_for_year: Optional[str] = None # Could be Decimal or float

@dataclass_json
@dataclass
class P60PayAndIncomeTaxDetails(BaseModel):
    pay: Optional[P60Pay] = None
    income_tax: Optional[P60IncomeTax] = None
    final_tax_code: Optional[str] = None

@dataclass_json
@dataclass
class P60NationalInsuranceContributionsInEmployment(BaseModel):
    nic_letter: Optional[str] = None
    earnings_at_lel: Optional[str] = None # Could be Decimal or float
    earnings_above_lel_to_pt: Optional[str] = None # Could be Decimal or float
    earnings_above_pt: Optional[str] = None # Could be Decimal or float

@dataclass_json
@dataclass
class P60NationalInsuranceContributions(BaseModel):
    in_this_employment: Optional[P60NationalInsuranceContributionsInEmployment] = None

@dataclass_json
@dataclass
class P60Model(BaseModel):
    document_type: Optional[str] = None
    tax_year_end: Optional[str] = None # Could be date like "5 April YYYY"
    employee_instructions: Optional[P60EmployeeInstructions] = None
    employee_details: Optional[P60EmployeeDetails] = None
    pay_and_income_tax_details: Optional[P60PayAndIncomeTaxDetails] = None
    national_insurance_contributions: Optional[P60NationalInsuranceContributions] = None

# Marriage certificate.json
@dataclass_json
@dataclass
class MarriageCertificateDetails(BaseModel):
    country: Optional[str] = None
    issuing_authority: Optional[str] = None
    certificate_type: Optional[str] = None
    registration_number: Optional[str] = None
    date_of_issue: Optional[str] = None # Could be date
    place_of_issue: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateTranslatorDetails(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    license_number: Optional[str] = None
    tax_id: Optional[str] = None
    languages: Optional[List[str]] = None
    certification_statement: Optional[str] = None
    appointment_authority: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateVerificationSeal(BaseModel):
    name: Optional[str] = None
    id: Optional[str] = None # Changed from id to seal_id to avoid Pydantic conflict if any
    validation_url: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateVerification(BaseModel):
    seal: Optional[MarriageCertificateVerificationSeal] = None
    qr_code: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateSpouse(BaseModel):
    given_names: Optional[str] = None
    family_name: Optional[str] = None
    tax_id: Optional[str] = None
    nationality: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateMarriagePlace(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateMarriageDetails(BaseModel):
    date_of_marriage: Optional[str] = None # Could be date
    place_of_marriage: Optional[MarriageCertificateMarriagePlace] = None

@dataclass_json
@dataclass
class MarriageCertificateWitness(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateOfficiant(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None

@dataclass_json
@dataclass
class MarriageCertificateModel(BaseModel):
    certificate_details: Optional[MarriageCertificateDetails] = None
    translator_details: Optional[MarriageCertificateTranslatorDetails] = None
    verification: Optional[MarriageCertificateVerification] = None
    spouse_1: Optional[MarriageCertificateSpouse] = None
    spouse_2: Optional[MarriageCertificateSpouse] = None
    marriage_details: Optional[MarriageCertificateMarriageDetails] = None
    witnesses: Optional[List[MarriageCertificateWitness]] = None
    officiant: Optional[MarriageCertificateOfficiant] = None

# DocList-Documents fo Sponsored Worker Visa.json
@dataclass_json
@dataclass
class DocListDocumentDetail(BaseModel):
    name: Optional[str] = None
    details: Optional[str] = None # Was null in example, can be string or other
    required_condition: Optional[str] = None

@dataclass_json
@dataclass
class DocListDocumentCategory(BaseModel):
    category: Optional[str] = None
    description: Optional[str] = None
    documents: Optional[List[DocListDocumentDetail]] = None

@dataclass_json
@dataclass
class DocListDocumentsForSponsoredWorkerVisaModel(BaseModel):
    visa_type: Optional[str] = None
    document_categories: Optional[List[DocListDocumentCategory]] = None
    source_information: Optional[str] = None

# Apartment invoice.json
@dataclass_json
@dataclass
class ApartmentInvoiceToAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    postcode: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceTo(BaseModel):
    name: Optional[str] = None
    address: Optional[ApartmentInvoiceToAddress] = None

@dataclass_json
@dataclass
class ApartmentInvoiceIssuerAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    postcode: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceIssuerContact(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceIssuer(BaseModel):
    name: Optional[str] = None
    address: Optional[ApartmentInvoiceIssuerAddress] = None
    contact: Optional[ApartmentInvoiceIssuerContact] = None

@dataclass_json
@dataclass
class ApartmentInvoicePropertyAddress(BaseModel):
    apartment: Optional[str] = None
    building: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    postcode: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceProperty(BaseModel):
    address: Optional[ApartmentInvoicePropertyAddress] = None

@dataclass_json
@dataclass
class ApartmentInvoiceTenancyDetails(BaseModel):
    term: Optional[str] = None
    rental_amount_per_week: Optional[str] = None # Could be Decimal or float
    total_rental_first_instalment: Optional[str] = None # Could be Decimal or float

@dataclass_json
@dataclass
class ApartmentInvoiceVat(BaseModel):
    rate: Optional[str] = None # Could be percentage or float
    amount: Optional[str] = None # Could be Decimal or float

@dataclass_json
@dataclass
class ApartmentInvoicePaymentMethod(BaseModel):
    type: Optional[str] = None
    details: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceBankUkTransfer(BaseModel):
    sort_code: Optional[str] = None
    account_number: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceBankInternationalTransfers(BaseModel):
    iban: Optional[str] = None
    swift_bic: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceBankDetails(BaseModel):
    bank_name: Optional[str] = None
    account_name: Optional[str] = None
    uk_transfer: Optional[ApartmentInvoiceBankUkTransfer] = None
    international_transfers: Optional[ApartmentInvoiceBankInternationalTransfers] = None

@dataclass_json
@dataclass
class ApartmentInvoicePaymentInstructions(BaseModel):
    cleared_funds_required: Optional[str] = None
    accepted_payment_methods: Optional[List[ApartmentInvoicePaymentMethod]] = None
    bank_details: Optional[ApartmentInvoiceBankDetails] = None
    cash_payment_note: Optional[str] = None

@dataclass_json
@dataclass
class ApartmentInvoiceModel(BaseModel):
    invoice_date: Optional[str] = None # Could be date
    invoice_to: Optional[ApartmentInvoiceTo] = None
    issuer: Optional[ApartmentInvoiceIssuer] = None
    invoice_number: Optional[str] = None
    property: Optional[ApartmentInvoiceProperty] = None
    tenancy_details: Optional[ApartmentInvoiceTenancyDetails] = None
    deposit: Optional[str] = None # Could be Decimal or float
    administration_charge: Optional[str] = None # Could be Decimal or float
    vat: Optional[ApartmentInvoiceVat] = None
    amount_due: Optional[str] = None # Could be Decimal or float
    vat_number: Optional[str] = None
    payment_instructions: Optional[ApartmentInvoicePaymentInstructions] = None

# Consolidated Model
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ConsolidatedDocumentModelPartOne(BaseModel):
    utility_bill: Optional[UtilityBillModel] = None
    referee_info: Optional[RefereeInfoModel] = None
    referee_and_identity: Optional[RefereeAndIdentityModel] = None
    passport: Optional[PassportModel] = None
    parents_info: Optional[ParentsInfoModel] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ConsolidatedDocumentModelPartTwo(BaseModel):
    p60: Optional[P60Model] = None
    marriage_certificate: Optional[MarriageCertificateModel] = None
    doc_list_documents_for_sponsored_worker_visa: Optional[DocListDocumentsForSponsoredWorkerVisaModel] = None
    apartment_invoice: Optional[ApartmentInvoiceModel] = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ConsolidatedDocumentModel(BaseModel):
    partOne: Optional[ConsolidatedDocumentModelPartOne] = None
    partTwo: ConsolidatedDocumentModelPartTwo = None

