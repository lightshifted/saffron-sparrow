# Synthetic Medical Records Generator
## Overview

The Synthetic Medical Records Generator API provides an interface for generating synthetic medical records based on provided ICD (International Classification of Diseases) codes and CPT (Current Procedural Terminology) codes. This API leverages OpenAI API to generate text that simulates medical record entries, offering a valuable tool for research, training, and software testing within the healthcare industry. The API is built using FastAPI and integrates with Hugging Face's `datasets` library to access enriched medical data.

<details>
<summary>Example Output</summary>


```text
[Index] Primary Care Visit [Index]
Date: 04/12/2023 Time: 10:45 AM
Provider: Dr. _______, MD
Department: Family Medicine

Patient Information
Age: 34 years old
Gender: Female

Chief Complaint: "My nose won't stop running, and my eyes are so itchy."

History of Present Illness
Patient reports onset of nasal congestion and itchy eyes approximately 2 weeks ago which coincides with the start of the spring season. Symptoms seem to worsen when outdoors. No history of fever, cough, or other systemic symptoms. Patient has tried over-the-counter antihistamines with minimal relief.

Allergies: No known drug allergies. Seasonal allergies to pollen.

Past Medical History
- Seasonal allergic rhinitis

Current Medications: 
- OTC Cetirizine 10 mg daily

Review of Systems
Allergic/Immunologic: Reports seasonal allergies. Negative for food allergies.
ENT: Reports nasal congestion, itchy eyes, and sneezing. Negative for sore throat, ear pain, or hearing loss.
Respiratory: Negative for shortness of breath, wheezing, or cough.
The rest of the systems review is unremarkable.

Physical Examination
Vital Signs: BP 120/78 mmHg, Heart Rate 72 bpm, Temp 98.6 °F, Resp 16/min
ENT: Nasal mucosa swollen and pale, clear nasal discharge, no sinus tenderness, conjunctivae are mildly erythematous and edematous.
Lungs: Clear to auscultation bilaterally. No wheezes, crackles, or rhonchi.
The remainder of the physical exam is within normal limits.

Assessment/Plan
- Diagnosis: Allergic rhinitis, exacerbated by pollen exposure.
- Continue daily antihistamine (Cetirizine 10 mg daily). Consider switching to a different antihistamine if symptoms persist.
- Start intranasal corticosteroid (Fluticasone propionate 50 mcg/spray, one spray in each nostril daily) for better control of nasal symptoms.
- Consider adding over-the-counter artificial tears to alleviate itchy and red eyes.
- Patient educated on avoiding outdoor activities during high pollen days and keeping windows closed to minimize exposure.
- Follow-up: Return to clinic in 4 weeks for symptom reevaluation or sooner if symptoms worsen.

Dr. _______, MD
Family Medicine
04/12/2023
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
  "model": "gpt-3.5-turbo-1106",
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
    "model":"gpt-3.5-turbo-1106",
    "messages": [
        {
            "role": "user",
            "content": "J30.1"
        }
    ], 
    "cpt_codes": ["76830"]
}'
```