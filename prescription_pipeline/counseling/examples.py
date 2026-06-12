from __future__ import annotations

import dspy


def build_training_examples() -> list[dspy.Example]:
    """Twenty grounded patient-counseling examples for BootstrapFewShot (plan Phase 5.3)."""
    raw = [
        {
            "database_facts": (
                '{"drug": "amoxicillin 500mg", "form": "capsule", '
                '"how_to_use": "Take as advised. Swallow whole.", '
                '"food_instruction": "either", '
                '"side_effects": "Nausea, diarrhea may occur.", '
                '"safety_advice": "Tell your doctor if you are pregnant."}'
            ),
            "ocr_variables": (
                '{"drug_name": "amox-500", "dosage": "500mg", '
                '"frequency": "TDS", "duration": "5 days", "match_status": "matched"}'
            ),
            "patient_query": "Can I take this with food?",
            "patient_response": (
                "Your prescription shows amoxicillin 500 mg three times daily for 5 days. "
                "The medicine information says it may be taken with or without food. "
                "If you feel stomach upset, taking it with a meal may help. "
                "Ask your doctor or pharmacist if you are unsure."
            ),
        },
        {
            "database_facts": (
                '{"drug": "metformin 500mg", "form": "tablet", '
                '"food_instruction": "with_food", '
                '"side_effects": "Nausea and stomach upset are common initially.", '
                '"safety_advice": "Avoid heavy alcohol use."}'
            ),
            "ocr_variables": (
                '{"drug_name": "metformin", "dosage": "500mg", '
                '"frequency": "BD", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Why do I feel sick after starting this?",
            "patient_response": (
                "Your prescription lists metformin 500 mg twice daily. "
                "The database notes nausea and stomach upset are common when starting. "
                "It also recommends taking it with food. "
                "Contact your doctor if symptoms are severe or do not improve."
            ),
        },
        {
            "database_facts": (
                '{"drug": "azithromycin 200mg", "form": "suspension", '
                '"how_to_use": "Shake well. Measure with the cup provided.", '
                '"food_instruction": "either", '
                '"side_effects": "Diarrhea, abdominal pain may occur."}'
            ),
            "ocr_variables": (
                '{"drug_name": "azithrail", "dosage": "3.5ml", '
                '"frequency": "once a day", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Can I mix it in juice?",
            "patient_response": (
                "Your prescription shows azithromycin suspension once daily. "
                "The instructions say to shake well and measure with the provided cup. "
                "The information provided does not specifically mention mixing with juice. "
                "Please confirm with your pharmacist before mixing with other drinks."
            ),
        },
        {
            "database_facts": (
                '{"drug": "paracetamol 500mg", "form": "tablet", '
                '"how_to_use": "Take as directed. Do not exceed recommended dose.", '
                '"safety_advice": "Use caution with liver disease."}'
            ),
            "ocr_variables": (
                '{"drug_name": "pcm", "dosage": "500mg", '
                '"frequency": "SOS", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "How often can I take it for fever?",
            "patient_response": (
                "Your prescription shows paracetamol 500 mg as needed (SOS). "
                "The database says to take as directed and not exceed the recommended dose. "
                "I do not have the exact maximum daily dose in the provided facts. "
                "Please follow your doctor's instructions or ask your pharmacist."
            ),
        },
        {
            "database_facts": (
                '{"drug": "levosalbutamol", "form": "nebulisation", '
                '"how_to_use": "Use only as prescribed for nebulisation.", '
                '"side_effects": "Tremor, fast heartbeat may occur."}'
            ),
            "ocr_variables": (
                '{"drug_name": "levosalbutamol", "dosage": "", '
                '"frequency": "nebulise", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Is it normal to feel shaky after nebulisation?",
            "patient_response": (
                "Your prescription includes levosalbutamol for nebulisation. "
                "The database lists tremor and fast heartbeat as possible side effects. "
                "Mild shakiness can occur for some people. "
                "Seek urgent care if breathing worsens or symptoms are severe."
            ),
        },
        {
            "database_facts": '{"note": "No knowledge-base match available.", "ocr_drug_name": "vitb2"}',
            "ocr_variables": (
                '{"drug_name": "vitb2", "dosage": "0.75mg", '
                '"frequency": "once a day", "duration": "1 month", "match_status": "skipped"}'
            ),
            "patient_query": "What is this vitamin for?",
            "patient_response": (
                "Your prescription shows vitamin B2 0.75 mg once daily for one month. "
                "I do not have verified database information about this item in the provided context. "
                "Please ask your prescribing doctor why it was prescribed for you."
            ),
        },
        {
            "database_facts": (
                '{"drug": "lactulose 10gm", "form": "syrup", '
                '"how_to_use": "Take once daily as advised.", '
                '"side_effects": "Bloating and gas may occur."}'
            ),
            "ocr_variables": (
                '{"drug_name": "lactulose", "dosage": "10ml", '
                '"frequency": "once a day", "duration": "1 month", "match_status": "matched"}'
            ),
            "patient_query": "When will it start working?",
            "patient_response": (
                "Your prescription lists lactulose 10 ml once daily for one month. "
                "The provided database facts do not state how quickly it works. "
                "Bloating and gas are listed as possible side effects. "
                "Contact your doctor if constipation does not improve."
            ),
        },
        {
            "database_facts": (
                '{"drug": "fexofenadine 30mg", "form": "syrup", '
                '"food_instruction": "either", '
                '"side_effects": "Headache, drowsiness may occur."}'
            ),
            "ocr_variables": (
                '{"drug_name": "allegra", "dosage": "5ml", '
                '"frequency": "twice a day", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Can I drive after taking this?",
            "patient_response": (
                "Your prescription shows fexofenadine syrup 5 ml twice daily. "
                "The database mentions headache and drowsiness as possible side effects. "
                "Do not drive until you know how this medicine affects you. "
                "Ask your doctor if you feel unusually sleepy."
            ),
        },
        {
            "database_facts": (
                '{"drug": "iron 30mg", "form": "suspension", '
                '"food_instruction": "empty_stomach", '
                '"quick_tips": "Iron-rich foods may support treatment."}'
            ),
            "ocr_variables": (
                '{"drug_name": "elemental iron", "dosage": "5ml", '
                '"frequency": "once a day", "duration": "3 months", "match_status": "matched"}'
            ),
            "patient_query": "Should I take it before breakfast?",
            "patient_response": (
                "Your prescription shows iron suspension 5 ml once daily for three months. "
                "The database suggests taking on an empty stomach when possible. "
                "If it upsets your stomach, ask your doctor whether to take it with a small snack."
            ),
        },
        {
            "database_facts": (
                '{"drug": "terbutaline 1.25mg", "form": "liquid", '
                '"safety_advice": "Use caution in pregnancy. Consult your doctor.", '
                '"side_effects": "Tremor, increased heart rate."}'
            ),
            "ocr_variables": (
                '{"drug_name": "terbutaline", "dosage": "1.25mg", '
                '"frequency": "", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Is this safe in pregnancy?",
            "patient_response": (
                "Your prescription includes terbutaline 1.25 mg. "
                "The database advises caution in pregnancy and to consult your doctor. "
                "I cannot confirm safety for your situation from the provided facts alone. "
                "Please speak with your obstetrician before continuing."
            ),
        },
        {
            "database_facts": (
                '{"drug": "ambroxol 30mg", "form": "syrup", '
                '"how_to_use": "Take as directed.", "side_effects": "Nausea, diarrhea."}'
            ),
            "ocr_variables": (
                '{"drug_name": "ambrolite", "dosage": "5ml", '
                '"frequency": "twice a day", "duration": "3 days", "match_status": "matched"}'
            ),
            "patient_query": "Can I stop early if cough improves?",
            "patient_response": (
                "Your prescription shows ambroxol syrup twice daily for three days. "
                "The provided facts do not say whether you may stop early. "
                "Do not stop or change the course without asking your prescribing doctor."
            ),
        },
        {
            "database_facts": (
                '{"drug": "polyethylene glycol 17g", "form": "sachet", '
                '"how_to_use": "Mix in water or juice and drink once daily."}'
            ),
            "ocr_variables": (
                '{"drug_name": "pegura", "dosage": "17g", '
                '"frequency": "once a day", "duration": "1 month", "match_status": "matched"}'
            ),
            "patient_query": "How much water should I mix it with?",
            "patient_response": (
                "Your prescription lists polyethylene glycol 17 g once daily for one month. "
                "The database says to mix in water or juice and drink once daily, "
                "but does not specify the exact water volume here. "
                "Follow the packet instructions or ask your pharmacist."
            ),
        },
        {
            "database_facts": (
                '{"drug": "folic acid 550mcg", "form": "suspension", '
                '"how_to_use": "Take once daily as advised."}'
            ),
            "ocr_variables": (
                '{"drug_name": "folic acid", "dosage": "5ml", '
                '"frequency": "once a day", "duration": "3 months", "match_status": "matched"}'
            ),
            "patient_query": "Do I need this with iron?",
            "patient_response": (
                "Your prescription shows folic acid once daily for three months. "
                "The provided database facts do not describe combinations with other medicines. "
                "Please ask your doctor why both were prescribed together."
            ),
        },
        {
            "database_facts": (
                '{"drug": "multivitamin", "form": "syrup", '
                '"how_to_use": "Take once daily.", "side_effects": "Generally well tolerated."}'
            ),
            "ocr_variables": (
                '{"drug_name": "zimcovit", "dosage": "5ml", '
                '"frequency": "once a day", "duration": "1 month", "match_status": "matched"}'
            ),
            "patient_query": "Can children take the same dose?",
            "patient_response": (
                "Your prescription shows a multivitamin syrup 5 ml once daily for one month. "
                "The provided facts do not include child dosing. "
                "Do not give this to a child unless your pediatrician confirms the dose."
            ),
        },
        {
            "database_facts": (
                '{"drug": "amoxicillin 500mg", "form": "capsule", '
                '"safety_advice": "Complete the full course unless your doctor says otherwise."}'
            ),
            "ocr_variables": (
                '{"drug_name": "amoxicillin", "dosage": "500mg", '
                '"frequency": "TDS", "duration": "5 days", "match_status": "matched"}'
            ),
            "patient_query": "I feel better after 2 days. Can I stop?",
            "patient_response": (
                "Your prescription is amoxicillin 500 mg three times daily for 5 days. "
                "The database advises completing the full course unless your doctor says otherwise. "
                "Stopping early can allow infection to return. "
                "Call your doctor before stopping."
            ),
        },
        {
            "database_facts": (
                '{"drug": "metformin 500mg", "form": "tablet", '
                '"safety_advice": "Avoid heavy alcohol use."}'
            ),
            "ocr_variables": (
                '{"drug_name": "metformin", "dosage": "500mg", '
                '"frequency": "BD", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Can I drink alcohol tonight?",
            "patient_response": (
                "Your prescription lists metformin 500 mg twice daily. "
                "The database advises avoiding heavy alcohol use. "
                "I cannot tell you whether a small amount is safe for you. "
                "Please check with your doctor or pharmacist."
            ),
        },
        {
            "database_facts": (
                '{"drug": "paracetamol 500mg", "form": "tablet", '
                '"side_effects": "Rare serious liver injury with overdose."}'
            ),
            "ocr_variables": (
                '{"drug_name": "paracetamol", "dosage": "500mg", '
                '"frequency": "SOS", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Can I take it with my other pain tablet?",
            "patient_response": (
                "Your prescription includes paracetamol 500 mg as needed. "
                "The provided facts do not list your other medicines or interactions. "
                "Do not combine pain medicines without pharmacist or doctor approval."
            ),
        },
        {
            "database_facts": (
                '{"drug": "azithromycin 200mg", "form": "suspension", '
                '"side_effects": "Diarrhea may occur."}'
            ),
            "ocr_variables": (
                '{"drug_name": "azithromycin", "dosage": "3.5ml", '
                '"frequency": "once a day", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "What should I do if I get diarrhea?",
            "patient_response": (
                "Your prescription shows azithromycin suspension once daily. "
                "Diarrhea is listed as a possible side effect. "
                "Stay hydrated and contact your doctor if diarrhea is severe, bloody, "
                "or does not improve."
            ),
        },
        {
            "database_facts": (
                '{"drug": "fexofenadine 30mg", "form": "syrup", '
                '"food_instruction": "either"}'
            ),
            "ocr_variables": (
                '{"drug_name": "fexofenadine", "dosage": "5ml", '
                '"frequency": "BD", "duration": "", "match_status": "matched"}'
            ),
            "patient_query": "Can I take antacids with it?",
            "patient_response": (
                "Your prescription lists fexofenadine syrup twice daily. "
                "The provided database facts do not mention antacids or interactions. "
                "Ask your pharmacist before combining with antacids or other medicines."
            ),
        },
        {
            "database_facts": (
                '{"drug": "lactulose 10gm", "form": "syrup", '
                '"side_effects": "Bloating and gas."}'
            ),
            "ocr_variables": (
                '{"drug_name": "lactulose", "dosage": "10ml", '
                '"frequency": "OD", "duration": "1 month", "match_status": "matched"}'
            ),
            "patient_query": "Will it cause cramps?",
            "patient_response": (
                "Your prescription shows lactulose 10 ml once daily for one month. "
                "The database lists bloating and gas as possible side effects. "
                "Abdominal cramps are not listed in the provided facts. "
                "Tell your doctor if cramps are bothersome or persistent."
            ),
        },
    ]

    examples: list[dspy.Example] = []
    for row in raw:
        example = dspy.Example(**row).with_inputs(
            "database_facts",
            "ocr_variables",
            "patient_query",
        )
        examples.append(example)
    return examples
