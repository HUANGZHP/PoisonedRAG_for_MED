"""
utils.py — Common utilities for building triplet training data for Contriever.

Provides:
- Data loading for PubMedQA and MedQA datasets
- Medical entity replacement dictionaries
- Text normalization and validation helpers
- Quality checking for generated negatives
"""

import json
import random
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def setup_logger(verbose: bool = True) -> None:
    """Configure the root logger with a stream handler.

    Default level is INFO so that progress messages are visible.
    Pass verbose=False to suppress to WARNING.
    """
    level = logging.INFO if verbose else logging.INFO  # Always INFO for pipeline
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ===================================================================
# Medical Entity Replacement Dictionaries
# ===================================================================

# Disease replacements:  (original_substring → misleading_replacement)
DISEASE_REPLACEMENTS: Dict[str, str] = {
    # Cardiovascular
    "hypertension": "hypotension",
    "myocardial infarction": "stable angina",
    "heart failure": "atrial fibrillation",
    "coronary artery disease": "pericarditis",
    "stroke": "transient ischemic attack",
    "atherosclerosis": "vasculitis",
    "deep vein thrombosis": "superficial thrombophlebitis",
    "pulmonary embolism": "pneumonia",
    # Infectious
    "pneumonia": "bronchitis",
    "tuberculosis": "sarcoidosis",
    "meningitis": "encephalitis",
    "sepsis": "systemic inflammatory response syndrome",
    "urinary tract infection": "interstitial cystitis",
    "cellulitis": "lymphedema",
    "endocarditis": "rheumatic heart disease",
    "osteomyelitis": "osteoarthritis",
    # Oncology
    "carcinoma": "adenoma",
    "lymphoma": "sarcoidosis",
    "leukemia": "myelodysplastic syndrome",
    "melanoma": "seborrheic keratosis",
    "glioblastoma": "meningioma",
    "breast cancer": "fibroadenoma",
    "lung cancer": "pulmonary tuberculosis",
    "colorectal cancer": "irritable bowel syndrome",
    "prostate cancer": "benign prostatic hyperplasia",
    # Endocrine
    "diabetes mellitus": "diabetes insipidus",
    "hyperthyroidism": "hypothyroidism",
    "Cushing syndrome": "Addison disease",
    "acromegaly": "growth hormone deficiency",
    # Neurology
    "Alzheimer disease": "Parkinson disease",
    "multiple sclerosis": "amyotrophic lateral sclerosis",
    "epilepsy": "syncope",
    "migraine": "tension headache",
    # Gastroenterology
    "Crohn disease": "ulcerative colitis",
    "cirrhosis": "hepatic steatosis",
    "pancreatitis": "cholecystitis",
    "gastroesophageal reflux": "eosinophilic esophagitis",
    # Respiratory
    "asthma": "COPD",
    "pulmonary fibrosis": "hypersensitivity pneumonitis",
    # Renal
    "acute kidney injury": "chronic kidney disease",
    "nephrotic syndrome": "nephritic syndrome",
    # Obstetrics / Gynecology
    "preeclampsia": "gestational hypertension",
    "endometriosis": "pelvic inflammatory disease",
    "polycystic ovary syndrome": "primary ovarian insufficiency",
    # Pediatrics
    "cystic fibrosis": "primary ciliary dyskinesia",
    "Hirschsprung disease": "chronic idiopathic constipation",
    # Rheumatology
    "rheumatoid arthritis": "osteoarthritis",
    "systemic lupus erythematosus": "mixed connective tissue disease",
    "gout": "pseudogout",
    "ankylosing spondylitis": "mechanical low back pain",
    # Psychiatry
    "major depressive disorder": "generalized anxiety disorder",
    "bipolar disorder": "borderline personality disorder",
    "schizophrenia": "schizoaffective disorder",
}

# Drug / medication replacements
DRUG_REPLACEMENTS: Dict[str, str] = {
    # Antibiotics
    "penicillin": "amoxicillin",
    "ceftriaxone": "cefazolin",
    "vancomycin": "linezolid",
    "azithromycin": "clarithromycin",
    "doxycycline": "minocycline",
    "ciprofloxacin": "levofloxacin",
    "metronidazole": "clindamycin",
    "nitrofurantoin": "trimethoprim-sulfamethoxazole",
    "gentamicin": "tobramycin",
    # Cardiovascular
    "lisinopril": "enalapril",
    "losartan": "valsartan",
    "metoprolol": "atenolol",
    "amlodipine": "nifedipine",
    "furosemide": "hydrochlorothiazide",
    "spironolactone": "eplerenone",
    "atorvastatin": "rosuvastatin",
    "warfarin": "dabigatran",
    "apixaban": "rivaroxaban",
    "clopidogrel": "ticagrelor",
    "digoxin": "ivabradine",
    "nitroglycerin": "isosorbide mononitrate",
    # Diabetes
    "metformin": "glipizide",
    "insulin": "sitagliptin",
    "glargine": "dapagliflozin",
    # Neurology / Psychiatry
    "sertraline": "fluoxetine",
    "fluoxetine": "citalopram",
    "escitalopram": "paroxetine",
    "venlafaxine": "duloxetine",
    "haloperidol": "risperidone",
    "risperidone": "olanzapine",
    "olanzapine": "quetiapine",
    "levodopa": "pramipexole",
    "carbamazepine": "lamotrigine",
    "valproate": "topiramate",
    "phenytoin": "levetiracetam",
    # Analgesics
    "morphine": "oxycodone",
    "ibuprofen": "naproxen",
    "acetaminophen": "aspirin",
    "tramadol": "codeine",
    # Endocrine
    "levothyroxine": "methimazole",
    "prednisone": "dexamethasone",
    # Oncology
    "tamoxifen": "raloxifene",
    "methotrexate": "azathioprine",
    "cyclophosphamide": "mycophenolate mofetil",
    # GI
    "omeprazole": "pantoprazole",
    "ranitidine": "famotidine",
    "ondansetron": "metoclopramide",
    # Respiratory
    "albuterol": "ipratropium",
    "fluticasone": "budesonide",
    "montelukast": "zafirlukast",
}

# Treatment / procedure replacements
TREATMENT_REPLACEMENTS: Dict[str, str] = {
    "appendectomy": "cholecystectomy",
    "cholecystectomy": "appendectomy",
    "coronary artery bypass graft": "percutaneous coronary intervention",
    "percutaneous coronary intervention": "coronary artery bypass graft",
    "total hip arthroplasty": "hip hemiarthroplasty",
    "total knee arthroplasty": "unicompartmental knee arthroplasty",
    "cesarean section": "vacuum-assisted vaginal delivery",
    "colectomy": "ileostomy",
    "lobectomy": "segmentectomy",
    "mastectomy": "lumpectomy",
    "radical prostatectomy": "transurethral resection of the prostate",
    "hemodialysis": "peritoneal dialysis",
    "mechanical ventilation": "noninvasive positive pressure ventilation",
    "cardiopulmonary resuscitation": "synchronized cardioversion",
    "radiation therapy": "chemotherapy",
    "chemotherapy": "immunotherapy",
    "bone marrow transplantation": "stem cell support",
    "splenectomy": "partial splenic embolization",
    "thyroidectomy": "radioactive iodine ablation",
    "laminectomy": "microdiscectomy",
    "endoscopic retrograde cholangiopancreatography": "magnetic resonance cholangiopancreatography",
    "laparoscopic surgery": "open surgery",
    "angioplasty": "stent placement",
}

# Diagnostic method replacements
DIAGNOSTIC_REPLACEMENTS: Dict[str, str] = {
    "computed tomography": "magnetic resonance imaging",
    "CT scan": "MRI",
    "magnetic resonance imaging": "computed tomography",
    "MRI": "CT scan",
    "ultrasound": "X-ray",
    "X-ray": "ultrasound",
    "electrocardiogram": "echocardiogram",
    "echocardiogram": "electrocardiogram",
    "colonoscopy": "flexible sigmoidoscopy",
    "upper endoscopy": "barium swallow",
    "biopsy": "fine needle aspiration",
    "lumbar puncture": "CT myelography",
    "mammography": "breast ultrasound",
    "bronchoscopy": "thoracentesis",
    "PET scan": "bone scan",
    "angiography": "Doppler ultrasound",
    "electroencephalogram": "polysomnography",
    "pulmonary function test": "arterial blood gas",
}

# Biomarker / lab value replacements
BIOMARKER_REPLACEMENTS: Dict[str, str] = {
    "troponin": "CK-MB",
    "BNP": "ANP",
    "creatinine": "urea",
    "ALT": "AST",
    "AST": "GGT",
    "alkaline phosphatase": "acid phosphatase",
    "hemoglobin A1c": "fasting glucose",
    "TSH": "free T4",
    "procalcitonin": "C-reactive protein",
    "D-dimer": "fibrinogen",
    "PSA": "free testosterone",
    "CA-125": "CEA",
    "rheumatoid factor": "anti-CCP antibody",
    "ANA": "anti-dsDNA",
    "lactate": "pyruvate",
    "amylase": "lipase",
}

# Outcome replacements
OUTCOME_REPLACEMENTS: Dict[str, str] = {
    "mortality": "hospital readmission",
    "overall survival": "quality of life",
    "progression-free survival": "overall response rate",
    "complete remission": "partial response",
    "relapse": "secondary malignancy",
    "adverse event": "therapeutic benefit",
    "treatment failure": "drug intolerance",
    "cure": "palliation",
    "significant improvement": "marginal benefit",
}

# Combined lookup for replacement
_all_replacements: Dict[str, str] = {}
_all_replacements.update(DISEASE_REPLACEMENTS)
_all_replacements.update(DRUG_REPLACEMENTS)
_all_replacements.update(TREATMENT_REPLACEMENTS)
_all_replacements.update(DIAGNOSTIC_REPLACEMENTS)
_all_replacements.update(BIOMARKER_REPLACEMENTS)
_all_replacements.update(OUTCOME_REPLACEMENTS)

# Sort replacements by length (longest first) for greedy matching
SORTED_REPLACEMENT_KEYS: List[str] = sorted(
    _all_replacements.keys(),
    key=lambda k: len(k),
    reverse=True,
)


# ===================================================================
# Data Loading
# ===================================================================

def load_pubmedqa(path: str) -> List[Dict[str, Any]]:
    """Load PubMedQA data from a JSON file.

    Expected format (from official_pqal):
        { "PMID": { "QUESTION": ..., "CONTEXTS": [...], "final_decision": ..., "LONG_ANSWER": ... }, ... }

    Returns:
        List of dicts with keys "query", "positive", "id", "meta".
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples: List[Dict[str, Any]] = []
    for pid, record in raw.items():
        question = (record.get("QUESTION") or "").strip()
        contexts = record.get("CONTEXTS") or []
        positive = " ".join(str(c).strip() for c in contexts if c).strip()

        if not question or not positive:
            continue

        samples.append({
            "id": pid,
            "query": question,
            "positive": positive,
            "meta": {
                "final_decision": record.get("final_decision", ""),
                "long_answer": record.get("LONG_ANSWER", ""),
                "year": record.get("YEAR", ""),
            },
        })
    return samples


def load_medqa(path: str) -> List[Dict[str, Any]]:
    """Load MedQA data from a JSONL file.

    Expected format (from phrases_no_exclude):
        { "question": ..., "answer": ..., "options": {...}, "answer_idx": ... }

    Since no explanation/context is available, positive = question + correct answer.

    Returns:
        List of dicts with keys "query", "positive", "id", "meta".
    """
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            question = (record.get("question") or "").strip()
            answer = (record.get("answer") or "").strip()
            answer_idx = (record.get("answer_idx") or "").strip()
            options = record.get("options") or {}

            if not question or not answer:
                continue

            positive = f"{question} {answer}"

            samples.append({
                "id": f"medqa_{line_idx}",
                "query": question,
                "positive": positive,
                "meta": {
                    "answer": answer,
                    "answer_idx": answer_idx,
                    "options": options,
                    "meta_info": record.get("meta_info", ""),
                },
            })
    return samples


# ===================================================================
# Text Normalization & Validation
# ===================================================================

def normalize_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return " ".join(str(text or "").split())


def check_quality(
    query: str,
    positive: str,
    negative: str,
    tolerance: float = 0.20,
) -> Tuple[bool, str]:
    """Validate a triplet sample.

    Returns:
        (ok, reason) — True if all checks pass.
    """
    if not query:
        return False, "empty query"
    if not positive:
        return False, "empty positive"
    if not negative:
        return False, "empty negative"

    query_n = normalize_text(query)
    positive_n = normalize_text(positive)
    negative_n = normalize_text(negative)

    if negative_n == positive_n:
        return False, "negative equals positive"

    # Length check (±tolerance)
    len_pos = len(positive_n.split())
    len_neg = len(negative_n.split())
    if len_pos == 0:
        return False, "zero-length positive"
    ratio = len_neg / len_pos
    if ratio < (1 - tolerance) or ratio > (1 + tolerance):
        return False, (
            f"length mismatch: positive={len_pos} words, "
            f"negative={len_neg} words, ratio={ratio:.2f}"
        )

    return True, "ok"


# ===================================================================
# Medical Entity Extraction
# ===================================================================

def find_medical_entities(text: str) -> List[Tuple[str, int, int]]:
    """Find medical entities in text using the replacement dictionary.

    Returns:
        List of (entity_text, start_pos, end_pos) sorted by length descending.
    """
    text_lower = text.lower()
    entities: List[Tuple[str, int, int]] = []
    seen_positions: set = set()
    for key in SORTED_REPLACEMENT_KEYS:
        start = 0
        while True:
            idx = text_lower.find(key, start)
            if idx == -1:
                break
            # Check that this position hasn't been used by a longer match
            pos_range = range(idx, idx + len(key))
            if not any(p in seen_positions for p in pos_range):
                entities.append((key, idx, idx + len(key)))
                seen_positions.update(pos_range)
            start = idx + len(key)
    # Sort by position
    entities.sort(key=lambda x: x[1])
    return entities


# ===================================================================
# Query-guided keyword extraction
# ===================================================================

# Common English stopwords to filter out
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "and", "but", "or", "if", "because", "until", "while",
    "about", "this", "that", "these", "those", "it", "its", "which",
    "who", "whom", "what", "the", "their", "them", "they", "we", "you",
    "he", "she", "his", "her", "my", "your", "our", "any", "per",
    "patient", "patients", "study", "studies", "using", "used", "use",
    "also", "however", "therefore", "thus", "although", "whereas",
}


def extract_query_keywords(query: str, max_keywords: int = 10) -> List[str]:
    """Extract important medical keywords from a query.

    Filters out common stopwords and short words to identify
    disease names, drug names, procedures, etc.

    Args:
        query: The question text.
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of important lowercase keywords.
    """
    words = re.findall(r'[a-zA-Z]+', query.lower())
    keywords = []
    for w in words:
        if len(w) >= 4 and w not in _STOPWORDS:
            keywords.append(w)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:max_keywords]


def find_keyword_positions(text: str, keywords: List[str]) -> List[Tuple[str, int, int]]:
    """Find positions of query keywords in the text.

    Returns:
        List of (keyword, start, end) tuples.
    """
    text_lower = text.lower()
    results = []
    for kw in keywords:
        start = 0
        while True:
            idx = text_lower.find(kw, start)
            if idx == -1:
                break
            results.append((kw, idx, idx + len(kw)))
            start = idx + len(kw)
    results.sort(key=lambda x: x[1])
    return results


# ===================================================================
# Medical term patterns for regex-based extraction
# ===================================================================

# Common medical suffix/prefix patterns
MEDICAL_PATTERNS = [
    r'\b[A-Z][a-z]+ (?:disease|syndrome|disorder|deficiency|failure|insufficiency)\b',
    r'\b(?:acute|chronic|primary|secondary|metastatic|advanced|severe|mild|moderate) [a-z]+(?:itis|osis|emia|oma|opathy)\b',
    r'\b[a-z]+(?:itis|osis|emia|oma|opathy|plasia|penia|cytosis|ectasis|rrhage|rrhagia|rrhea|malacia|necrosis|sclerosis|stenosis|spasm|lysis|trophy|dystrophy)\b',
    r'\b(?:elevated|decreased|increased|reduced|high|low) [a-z]+\b',
    r'\b(?:serum|plasma|urinary|blood|CSF) [a-z]+\b',
    r'\b\d+(?:\.\d+)? (?:mg|mcg|g|mL|L|mmol|IU|units|percent|%)\b',
]


def extract_medical_terms_regex(text: str) -> List[Tuple[str, int, int]]:
    """Extract medical-sounding terms using regex patterns.

    This catches terms not in the hardcoded dictionaries.

    Returns:
        List of (term, start, end).
    """
    results = []
    seen = set()
    for pattern in MEDICAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            term = match.group()
            span = (match.start(), match.end())
            if span not in seen:
                results.append((term, span[0], span[1]))
                seen.add(span)
    results.sort(key=lambda x: x[1])
    return results


# ===================================================================
# Text Reconstruction Helper
# ===================================================================

def apply_replacements(
    text: str,
    replacements: List[Tuple[int, int, str]],
) -> str:
    """Apply position-based replacements to reconstruct text.

    Args:
        text: Original text.
        replacements: List of (start, end, new_string).

    Returns:
        Modified text.
    """
    # Sort by start position
    replacements = sorted(replacements, key=lambda x: x[0])
    result_parts: List[str] = []
    cursor = 0
    for start, end, new_str in replacements:
        result_parts.append(text[cursor:start])
        result_parts.append(new_str)
        cursor = end
    result_parts.append(text[cursor:])
    return "".join(result_parts)
