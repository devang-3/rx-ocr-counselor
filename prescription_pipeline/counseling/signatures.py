from __future__ import annotations

import dspy


class PatientCounselor(dspy.Signature):
    """Answer a patient question using ONLY database_facts and ocr_variables.

    Do not invent medical facts, dosages, interactions, or warnings.
    If the question cannot be answered from the provided context, say so and
    advise consulting the prescribing doctor or pharmacist.
    Use plain, empathetic language suitable for a patient.
    """

    database_facts: str = dspy.InputField(
        desc="Verified drug knowledge base fields (side effects, safety, how to use)",
    )
    ocr_variables: str = dspy.InputField(
        desc="Prescription fields extracted from OCR/NER (drug, dosage, frequency, duration)",
    )
    patient_query: str = dspy.InputField(desc="Patient question in plain language")
    patient_response: str = dspy.OutputField(desc="Grounded, empathetic patient-facing answer")
