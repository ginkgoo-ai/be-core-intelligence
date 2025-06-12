from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import date

class OtherName(BaseModel):
    givenName: str = ""
    familyName: str = ""

class TelephoneNumber(BaseModel):
    number: str = ""
    type: str = ""  # Enum: "Home", "Mobile", "Work"
    usageContext: str = ""
    contactPreference: str = ""  # Enum: "CallAndText", "CallOnly", "TextOnly", "NoContact"
    isPreferred: bool = True

class EmailAddress(BaseModel):
    emailAddress: str = ""
    isPrimaryContactEmail: bool = True
    isDecisionNotificationEmail: bool = True
    isPostDecisionCommunicationEmail: bool = True

class DurationAtAddress(BaseModel):
    years: int = 0
    months: int = 0

class HomeAddress(BaseModel):
    addressLine1: str = ""
    addressLine2: str = ""
    city: str = ""
    provinceOrState: str = ""
    postalCode: str = ""
    country: str = ""
    dateLivedFrom: str = "YYYY-MM"
    durationAtAddress: DurationAtAddress = DurationAtAddress()
    ownership: str = ""
    livingSituationDetails: str = ""

class ContactInformation(BaseModel):
    emails: List[EmailAddress] = Field(default_factory=list)
    telephoneNumbers: List[TelephoneNumber] = Field(default_factory=list)
    homeAddress: HomeAddress = HomeAddress()

class PersonalDetails(BaseModel):
    title: str = ""
    givenName: str = ""
    familyName: str = ""
    fullNameAsOnPassport: str = ""
    hasOtherNames: bool = False
    otherNames: List[OtherName] = Field(default_factory=list)
    sex: str = ""
    relationshipStatus: str = ""
    nationality: str = ""
    countryOfBirth: str = ""
    placeOfBirth: str = ""
    dateOfBirth: str = "YYYY-MM-DD"

class Passport(BaseModel):
    passportNumber: str = ""
    issuingAuthority: str = ""
    issueDate: str = "YYYY-MM-DD"
    expiryDate: str = "YYYY-MM-DD"

class BiometricResidencePermit(BaseModel):
    hasBRP: bool = False
    brpNumber: str = ""
    issueDate: str = "YYYY-MM-DD"
    expiryDate: str = "YYYY-MM-DD"

class NationalIdentityCard(BaseModel):
    hasNationalIdCard: bool = False
    idCardNumber: str = ""
    issuingAuthority: str = ""
    issueDate: str = "YYYY-MM-DD"
    expiryDate: str = "YYYY-MM-DD"

class IdentityDocuments(BaseModel):
    passport: Passport = Passport()
    biometricResidencePermit: BiometricResidencePermit = BiometricResidencePermit()
    nationalIdentityCard: NationalIdentityCard = NationalIdentityCard()

class SponsorAddress(BaseModel):
    addressLine1: str = ""
    addressLine2: str = ""
    city: str = ""
    postalCode: str = ""

class SponsorshipDetails(BaseModel):
    certificateOfSponsorshipNumber: str = ""
    sponsorLicenceNumber: str = ""
    sponsorName: str = ""
    sponsorAddress: SponsorAddress = SponsorAddress()
    jobTitle: str = ""
    socCode: str = ""
    grossAnnualSalary: int = 0
    workStartDate: str = "YYYY-MM-DD"
    workEndDate: str = "YYYY-MM-DD"

class DegreeDetails(BaseModel):
    degreeTitle: str = ""
    yearOfAward: str = ""
    countryOfAward: str = ""
    awardingInstitution: str = ""
    previousProofMethod: str = ""

class TestScores(BaseModel):
    reading: float = 0.0
    writing: float = 0.0
    speaking: float = 0.0
    listening: float = 0.0
    overall: float = 0.0

class TestDetails(BaseModel):
    provider: str = ""
    type: str = ""
    referenceNumber: str = ""
    dateOfTest: str = "YYYY-MM-DD"
    scores: TestScores = TestScores()

class ExemptionDetails(BaseModel):
    reason: str = ""
    professionalRegistrationNumber: str = ""

class EnglishLanguageAbility(BaseModel):
    methodOfProof: str = ""
    degreeDetails: DegreeDetails = DegreeDetails()
    testDetails: TestDetails = TestDetails()
    exemptionDetails: ExemptionDetails = ExemptionDetails()

class UKMedicalHistory(BaseModel):
    hasReceivedTreatment: bool = False
    treatmentDetails: str = ""
    wasTreatmentPaidFor: Optional[bool] = None

class PreviousUKVisa(BaseModel):
    visaType: str = ""
    issueDate: str = "YYYY-MM-DD"
    expiryDate: str = "YYYY-MM-DD"
    mainPurposeOfStay: str = ""
    conditions: str = ""

class KeyAreaVisit(BaseModel):
    hasVisited: bool = False
    numberOfVisitsLast10Years: int = 0

class SpecificTravelHistory(BaseModel):
    australia: KeyAreaVisit = KeyAreaVisit()
    canada: KeyAreaVisit = KeyAreaVisit()
    newZealand: KeyAreaVisit = KeyAreaVisit()
    usa: KeyAreaVisit = KeyAreaVisit()
    switzerland: KeyAreaVisit = KeyAreaVisit()
    eea: KeyAreaVisit = KeyAreaVisit()

class TravelCountry(BaseModel):
    countryName: str = ""
    reasonForVisit: str = ""
    entryDate: str = "YYYY-MM-DD"
    exitDate: str = "YYYY-MM-DD"

class TravelHistory(BaseModel):
    totalCountriesVisited: int = 0
    countries: List[TravelCountry] = Field(default_factory=list)

class BreachOrRefusal(BaseModel):
    hasDone: bool = False
    details: str = ""

class BreachesAndRefusals(BaseModel):
    enteredUkIllegally: BreachOrRefusal = BreachOrRefusal()
    remainedInUkBeyondVisa: BreachOrRefusal = BreachOrRefusal()
    breachedVisaConditionsInUk: BreachOrRefusal = BreachOrRefusal()
    deportedOrRemovedFromAnyCountry: BreachOrRefusal = BreachOrRefusal()
    refusedVisaForAnyCountry: BreachOrRefusal = BreachOrRefusal()
    refusedPermissionToStayInUk: BreachOrRefusal = BreachOrRefusal()
    refusedEntryAtUkBorder: BreachOrRefusal = BreachOrRefusal()

class ImmigrationHistory(BaseModel):
    ukMedicalHistory: UKMedicalHistory = UKMedicalHistory()
    hasBeenGrantedUkVisa: bool = False
    previousUkVisas: List[PreviousUKVisa] = Field(default_factory=list)
    specificTravelHistory: SpecificTravelHistory = SpecificTravelHistory()
    travelHistory: TravelHistory = TravelHistory()
    breachesAndRefusals: BreachesAndRefusals = BreachesAndRefusals()

class SpouseOrPartnerIdentityDocument(BaseModel):
    hasCurrentPassport: bool = False
    passportNumber: str = ""

class SpouseOrPartnerCurrentLocation(BaseModel):
    country: str = ""
    isInTheUK: bool = False

class SpouseOrPartner(BaseModel):
    isApplyingWithYou: bool = False
    givenName: str = ""
    familyName: str = ""
    hasOtherNames: bool = False
    otherNames: List[OtherName] = Field(default_factory=list)
    dateOfBirth: str = "YYYY-MM-DD"
    nationality: str = ""
    identityDocument: SpouseOrPartnerIdentityDocument = SpouseOrPartnerIdentityDocument()
    currentLocation: SpouseOrPartnerCurrentLocation = SpouseOrPartnerCurrentLocation()

class ChildIdentityDocument(BaseModel):
    hasCurrentPassport: bool = False
    passportNumber: str = ""

class Child(BaseModel):
    givenName: str = ""
    familyName: str = ""
    dateOfBirth: str = "YYYY-MM-DD"
    relationship: str = ""
    nationality: str = ""
    isApplyingWithYou: bool = False
    livesWithYouInUK: bool = False
    identityDocument: ChildIdentityDocument = ChildIdentityDocument()

class Parent(BaseModel):
    relation: str = ""
    givenName: str = ""
    familyName: str = ""
    dateOfBirth: str = "YYYY-MM-DD"
    nationality: str = ""

class FamilyDetails(BaseModel):
    spouseOrPartner: SpouseOrPartner = SpouseOrPartner()
    children: List[Child] = Field(default_factory=list)
    parents: List[Parent] = Field(default_factory=list)

class CriminalConviction(BaseModel):
    hasAny: bool = False
    details: str = ""

class WarCrimeOrTerrorism(BaseModel):
    hasBeenInvolved: bool = False
    details: str = ""

class SupportForTerroristOrganisation(BaseModel):
    hasGivenSupport: bool = False
    details: str = ""

class ExtremistViewsExpressed(BaseModel):
    hasExpressed: bool = False
    details: str = ""

class SensitiveEmployment(BaseModel):
    armedForces: CriminalConviction = CriminalConviction()
    government: CriminalConviction = CriminalConviction()
    media: CriminalConviction = CriminalConviction()
    securityOrganisations: CriminalConviction = CriminalConviction()
    judiciary: CriminalConviction = CriminalConviction()

class OtherInformation(BaseModel):
    hasInfoToDeclare: bool = False
    details: str = ""

class CharacterAndConduct(BaseModel):
    sensitiveEmployment: SensitiveEmployment = SensitiveEmployment()
    otherInformation: OtherInformation = OtherInformation()

class EmploymentHistoryItem(BaseModel):
    employerName: str = ""
    jobTitle: str = ""
    startDate: str = "YYYY-MM-DD"
    endDate: str = "YYYY-MM-DD"
    address: str = ""
    city: str = ""
    country: str = ""

class ConvictionsAndPenalties(BaseModel):
    criminalConvictions: CriminalConviction = CriminalConviction()
    drivingOffences: CriminalConviction = CriminalConviction()
    arrestsOrChargesAwaitingTrial: CriminalConviction = CriminalConviction()
    cautionsWarningsOrPenalties: CriminalConviction = CriminalConviction()
    civilCourtJudgements: CriminalConviction = CriminalConviction()
    ukImmigrationCivilPenalties: CriminalConviction = CriminalConviction()

class WarCrimesAndTerrorism(BaseModel):
    involvementInWarCrimes: WarCrimeOrTerrorism = WarCrimeOrTerrorism()
    involvementInTerroristActivities: WarCrimeOrTerrorism = WarCrimeOrTerrorism()
    supportForTerroristOrganisations: SupportForTerroristOrganisation = SupportForTerroristOrganisation()
    extremistViewsExpressed: ExtremistViewsExpressed = ExtremistViewsExpressed()

class AdditionalInformation(BaseModel):
    convictionsAndPenalties: ConvictionsAndPenalties = ConvictionsAndPenalties()
    warCrimesAndTerrorism: WarCrimesAndTerrorism = WarCrimesAndTerrorism()
    characterAndConduct: CharacterAndConduct = CharacterAndConduct()
    employmentHistory: List[EmploymentHistoryItem] = Field(default_factory=list)

class ApplicationDetails(BaseModel):
    intendedArrivalDate: str = "YYYY-MM-DD"
    visaLength: str = ""

class Profile(BaseModel):
    applicationDetails: ApplicationDetails = ApplicationDetails()
    personalDetails: PersonalDetails = PersonalDetails()
    contactInformation: ContactInformation = ContactInformation()
    identityDocuments: IdentityDocuments = IdentityDocuments()
    sponsorshipDetails: SponsorshipDetails = SponsorshipDetails()
    englishLanguageAbility: EnglishLanguageAbility = EnglishLanguageAbility()
    immigrationHistory: ImmigrationHistory = ImmigrationHistory()
    familyDetails: FamilyDetails = FamilyDetails()
    additionalInformation: AdditionalInformation = AdditionalInformation() 