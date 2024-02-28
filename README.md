# Synthetic Medical Records Generator
## Overview

The Synthetic Medical Records Generator API provides an interface for generating synthetic medical records based on provided ICD (International Classification of Diseases) codes and CPT (Current Procedural Terminology) codes. This API leverages OpenAI API to generate text that simulates medical record entries, offering a valuable tool for research, training, and software testing within the healthcare industry. The API is built using FastAPI and integrates with Hugging Face's `datasets` library to access enriched medical data.

<details>
<summary>Example Output</summary>


```text
[Index] Visit Summary [Index]
Visit Summary
Admitting Diagnosis: Allergic rhinitis due to pollen
Discharge Diagnosis: Same as admitting diagnosis
Patient is a 28-year-old with no significant past medical history presenting for evaluation of persistent nasal congestion, sneezing, and itchy eyes worsened during spring. Reports symptoms consistent with seasonal allergies, exacerbated by pollen exposure.

Discharge Exam:
BP: 120/78
Temp: 36.6 °C (97.9 °F)
Pulse: 72
Resp: 14
SpO2: 98%
Constitutional: No acute distress, well-developed, well-nourished
ENT: Nasal mucosa edematous, clear rhinorrhea noted. No nasal polyps.
Eyes: Conjunctival erythema, no discharge
Respiratory: Clear to auscultation bilaterally, no wheezes
Skin: No rash noted

Complications: None
Condition upon Discharge: Stable, symptoms improved with initial treatment

[Index] History and Physical [Index]
Physical Examination on Admission:
Constitutional: Alert, oriented, in no acute distress
ENT: Bilateral nasal congestion, mucosa swollen, clear discharge
Eyes: Watering, itchy, red conjunctiva
Respiratory: Breathing comfortably, no use of accessory muscles
Skin: No hives or eczematous changes

Allergies: No known drug allergies
Medications Upon Admission: None

Social History: Non-smoker. Works as an elementary school teacher. No pets at home. Reports increased outdoor activities during spring months.

Review of Systems:
Constitutional: Denies fever, weight loss
Respiratory: Reports sneezing, nasal congestion, occasional cough
ENT: Itchy throat, itchy eyes, clear rhinorrhea
Skin: Denies any rashes or itching

Assessment & Plan:
The patient presents with symptoms consistent with allergic rhinitis exacerbated by pollen exposure. Plan to start non-sedating antihistamine and intranasal corticosteroid spray for symptom control. Discussed avoidance of known allergens, use of air purifiers, and changing clothes after spending time outdoors during high pollen days. Follow-up in 2 weeks to reassess symptom control and adjust treatment as necessary.

Education provided regarding the identification and avoidance of trigger factors, importance of medication adherence, and possible side effects of prescribed medications.

Prescriptions Given at Discharge:
1. Fexofenadine 180 mg orally once daily
2. Fluticasone propionate nasal spray, 50 mcg per nostril once daily

Follow-Up:
Schedule follow-up appointment in 2 weeks with primary care physician to monitor progression and adjust medications if needed. 

Instructions for acute exacerbations include over-the-counter saline nasal spray for additional relief and contacting healthcare provider if symptoms worsen or if experiencing side effects of medications.

Electronically Signed by Dr. _____ at [date/time]
```
</details>

## Prerequisites
Before using this API, ensure the following requirements are met:
- Python 3.8+ is installed.
- An environment variable `HF_TOKEN` is set with your Hugging Face API token.
This token is necessary to access the datasets hosted on Hugging Face.
- An environment variable `OPENAI_API_KEY` is set with you OpenAI API key.

- Required Python libraries are installed, including `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `openai`, and `datasets`.

## API Configuration

### Environment Variables

- `HF_TOKEN`: Your Hugging Face API token used to authenticate and access datasets.
- `OPENAI_API_KEY`: Your OpenAI API key.

### CORS Middleware

The API is configured to accept requests from a specified origin for cross-origin resource sharing (CORS):
- **Allowed Origins**: The API accepts requests from the URL defined in the `HOME_URL` variable, facilitating requests from a web application hosted at this location.

## API Endpoints

### POST /api/generate/records

Generates synthetic medical records based on the provided ICD and CPT codes.

#### Request Body

- `model`: The identifier of the machine learning model to use for text generation. Default is set to `"gpt-3.5-turbo-1106"`.
- `messages`: A list of `Message` objects, where each object contains:
  - `role`: The role of the message sender (not actively used in the current implementation).
  - `content`: The content of the message, expected to be ICD codes for this implementation.
- `cpt_codes`: A list of CPT codes or a single CPT code string. These codes are used alongside ICD codes to generate the medical record.

#### Response

The response is a generated medical record text based on the provided codes and the selected model.

## Models

### Message

Represents a message in the chat request.

- `role`: A string indicating the role of the message sender.
- `content`: The content of the message, typically containing ICD codes in this context.

### ChatRequest

Defines the structure for the request to generate medical records.

- `model`: A string identifier for the machine learning model used for generation.
- `messages`: A list of `Message` objects.
- `cpt_codes`: Either a list of strings or a single string representing CPT codes.

## Usage

To use the API, send a POST request to `/api/generate/records` with a JSON body containing the required fields as defined in the `ChatRequest` model.

Example request body:
```json
{
  "model": "gpt-4-0125-preview",
  "messages": [
    {
      "role": "doctor",
      "content": "J18.9"
    }
  ],
  "cpt_codes": ["99213"]
}
```

Example curl call:
```bash
curl --location --request POST 'http://localhost:8000/api/generate/records' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--data-raw '{
    "model":"gpt-4-0125-preview",
    "messages": [
        {
            "role": "user",
            "content": "J30.1"
        }
    ], 
    "cpt_codes": ["76830"]
}'
```
