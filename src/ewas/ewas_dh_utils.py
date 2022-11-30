from strenum import StrEnum


class EWASDataHubDiseases(StrEnum):
    # Alzheimers = "Alzheimer's disease"
    # Asthma = "asthma"
    # AutismSpectrum = 'autism spectrum disorder'
    # ChildhoodAsthma = 'childhood asthma'
    Crohns = "Crohn's disease"
    Down = 'Down syndrome'
    # Graves = "Graves' disease"
    # Huntingtons = "Huntington's disease"
    CognitalAnomalies = 'intellectual disability and congenital anomalies'
    Kabuki = 'Kabuki syndrome'
    MS = 'multiple sclerosis'
    # NephrogenicRest = 'nephrogenic rest'
    Panic = 'panic disorder'
    Parkinsons = "Parkinson's disease"
    # Preeclampsia = 'preeclampsia'
    # Psoriasis = 'psoriasis'
    # RaspiratoryAllergy = 'respiratory allergy'
    # RheumatoidArthritis = 'rheumatoid arthritis'
    Schizophrenia = 'schizophrenia'
    SilverRussel = 'Silver Russell syndrome'
    Sjorgens = "Sjogren's syndrome"
    # Spina = 'spina bifida'
    Stroke = 'stroke'
    # InsulinResist = 'systemic insulin resistance'
    # Lupus = 'systemic lupus erythematosus'
    # Sclerosis = 'systemic sclerosis'
    T2D = 'type 2 diabetes'
    UlcerativeColitis = 'Ulcerative colitis'


class EWASDataHubTraits(StrEnum):
    Age = "age"
    Sex = "sex"
    BMI = "bmi"