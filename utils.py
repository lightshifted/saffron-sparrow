import json
import os
import random
from typing import List, Set, Union

import datasets
import numpy as np
import openai
import pandas as pd
import yaml
from dotenv import load_dotenv
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def get_code_details(code_list: Union[List[str], str], codebank: datasets.Dataset) -> datasets.Dataset:
    """
    Retrieves details for a list of codes from a given DataFrame.

    Parameters:
    code_list (list): A list of codes to retrieve details for.
    codebank (pd.DataFrame): A DataFrame containing code details with columns 'code' and 'code_details'.

    Returns:
    pd.DataFrame: A DataFrame with columns 'code' and 'details' containing the details for each code.
    """

    if not isinstance(codebank, pd.DataFrame):
        raise ValueError("codebank must be a pandas DataFrame")

    if "code" not in codebank.columns or "code_details" not in codebank.columns:
        raise ValueError("codebank DataFrame must contain 'code' and 'code_details' columns")

    if isinstance(code_list, str):
        code_list = code_list.split(", ")

    codeset = pd.DataFrame(columns=["code", "details"])
    for idx, code in enumerate(code_list):
        details_series = codebank["code_details"].loc[codebank["code"] == code]
        details = details_series.iloc[0] if not details_series.empty else "No details found"
        codeset.loc[idx] = [code, details]

    return codeset


def generate_random_int(low: int, high: int, exclude: int) -> int:
    """
    Generate a random integer within a specified range, excluding a specific number.

    Parameters:
    low (int): The lower bound of the range (inclusive).
    high (int): The upper bound of the range (exclusive).
    exclude (int): The integer to be excluded from the range.

    Returns:
    int: A randomly generated integer within the range [low, high) that is not equal to 'exclude'.

    """
    while True:
        result = np.random.randint(low, high)
        if result != exclude:
            return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat_completion(prompt: str, model: str = "gpt-3.5-turbo-1106") -> str:
    """
    Sends a chat completion request to the OpenAI API and retrieves the response.

    Parameters:
    prompt (str): The input prompt to be sent to the GPT-4 model.
    model (str): The OpenAI model to use. Defaults to gpt-4-1106-preview.

    Returns:
    str: The content of the response from the GPT-4 model.

    """
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])

    return json.loads(response.model_dump_json())


def filter_by_pdx(dataset: datasets.Dataset, pdxs: Union[str, Set[str]]) -> datasets.Dataset:
    """
    Filters a dataset based on a set of diagnosis codes (PDXs).

    Parameters:
        dataset (pd.DataFrame): The dataset to filter. Each record must have a 'px' column with codes separated by commas.
        pdxs (Union[str, Set[str]]): A string of comma-separated diagnosis codes or a set of diagnosis codes. The function filters the dataset to include records that contain any of these codes in their 'px' column.

    Returns:
        pd.DataFrame: A filtered dataset containing only records that match the provided diagnosis codes.
    """
    # Convert the pdxs string to a set of codes
    if isinstance(pdxs, str):
        pdxs = set(pdx.strip() for pdx in pdxs.split(","))

    def is_code_in_user_codes(row) -> bool:
        """Checks if the given row contains any of the user-specified diagnosis codes."""
        # Split the row codes into a set, handling None values
        row_codes = set() if row["px"] is None else set(row["px"].split(","))
        # Check for intersection between row codes and user codes
        return bool(row_codes.intersection(pdxs))

    # Filter the dataset using the above logic
    filtered_dataset = dataset.filter(is_code_in_user_codes)
    return filtered_dataset


def filter_by_cpt(dataset: datasets.Dataset, cpts: Union[str, Set[str]]) -> datasets.Dataset:
    """
    Filters a dataset based on a set of Current Procedural Terminology (CPT) codes.

    Parameters:
        dataset (pd.DataFrame): The dataset to be filtered, expected to have a 'cpt' column containing CPT codes.
        cpts (Union[str, Set[str]]): A string of comma-separated CPT codes or a set of CPT codes. The dataset is filtered to include records that contain any of these CPT codes in their 'cpt' column.

    Returns:
        pd.DataFrame: The filtered dataset containing only records that match the provided CPT codes.
    """
    # Convert the cpts string to a set of codes if it's a string
    if isinstance(cpts, str):
        cpts = set(cpt.strip() for cpt in cpts.split(","))

    def is_code_in_user_codes(row) -> bool:
        """Determines if the given row contains any of the user-specified CPT codes."""
        # Convert row's CPT codes into a set, handling None values
        row_codes = set() if row["cpt"] is None else set(row["cpt"].split(","))
        # Check for any common codes between the row's codes and the user-specified codes
        return bool(row_codes.intersection(cpts))

    # Filter the dataset using the specified logic
    filtered_dataset = dataset.filter(is_code_in_user_codes)
    return filtered_dataset


def select_random_rows(filtered_dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Selects two random rows from a given pandas DataFrame if it contains two or more rows.
    Returns the original DataFrame if it contains fewer than two rows.

    Parameters:
        filtered_dataset (datasets.Dataset): The dataset from which to select random rows.

    Returns:
        datasets.Dataset: A datasets Dataset object containing two randomly selected rows if the original dataset contains two or more rows,
        otherwise, the original dataset.
    """
    # Check if the dataset has 2 or more rows
    num_rows = len(filtered_dataset)
    if num_rows >= 2:
        # Generate 2 random indices
        random_indices = random.sample(range(num_rows), 2)
        # Select rows corresponding to these indices
        selected_rows = filtered_dataset.iloc[random_indices]
        return selected_rows
    else:
        # Return the original dataset if it has less than 2 rows
        return filtered_dataset


def filter_by_user_codes(dataset: datasets.Dataset, user_codes: Union[List[str], str]) -> datasets.Dataset:
    """
    Filters a dataset based on a list of user-specified codes.

    Parameters:
        dataset (datasets.Dataset): The dataset to be filtered.
        user_codes (Union[List[str], str]): A string of comma-separated user codes or a list of user code strings. The dataset is filtered to include records that contain any of these user codes in their 'code' key.

    Returns:
        datasets.Dataset: The filtered dataset, consisting of records that match the provided user codes.
    """
    # Split the user_codes string into a list and strip any whitespace
    if isinstance(user_codes, str):
        user_codes = [code.strip() for code in user_codes.split(",")]

    # Use the filter method of the dataset to find matching rows
    filtered_dataset = dataset.filter(lambda example: example["code"] in user_codes)

    return filtered_dataset


def generate(
    user_codes: list[str],
    codebank: datasets.Dataset,
    seed_data: datasets.Dataset,
    procedure_code: str = None,
    cpt_code: str = None,
    model: str = "gpt-3.5-turbo-1106",
):

    # Validate input types to ensure they match expected types.
    if not isinstance(codebank, datasets.Dataset):
        raise ValueError("codebank must be a Dataset object from the datasets library.")
    if not isinstance(seed_data, datasets.Dataset):
        raise ValueError("seed_data must be a Dataset object from the datasets library.")

    # Convert user_codes from list to string if not already in string format.
    if not isinstance(user_codes, str):
        user_codes = ", ".join(user_codes)

    # Split procedure and CPT codes strings into lists if they are not None or empty.
    if isinstance(procedure_code, str):
        procedure_code = procedure_code.split(", ")
    if isinstance(cpt_code, str):
        cpt_code = cpt_code.split(", ")

    # Reset empty strings to None for procedure and CPT codes.
    if procedure_code == "":
        procedure_code = None
    if cpt_code == "":
        cpt_code = None

    # Filter the codebank dataset by user codes to gather context data.
    context_data = filter_by_user_codes(codebank, user_codes)

    # Concatenate approximate synonyms from the filtered context data.
    approx_synonyms = ", ".join(context_data["approx_synonyms"])

    # Branch logic based on whether CPT codes are provided.
    if cpt_code is None:
        # If no CPT code is provided, select two random medical records for context.
        rand_val_1 = generate_random_int(0, seed_data.shape[0], 0)
        rand_val_2 = generate_random_int(0, seed_data.shape[0], rand_val_1)

        example_1 = seed_data[rand_val_1]
        example_2 = seed_data[rand_val_2]

        codes_1 = example_1["icd_codes"]
        codes_2 = example_2["icd_codes"]

        medical_record_1 = example_1["text"]
        medical_record_2 = example_2["text"]

        approx_synonyms_1 = filter_by_user_codes(codebank, codes_1)["approx_synonyms"]
        approx_synonyms_2 = filter_by_user_codes(codebank, codes_2)["approx_synonyms"]

    else:
        # If CPT codes are provided, filter the seed data by these codes and select random samples.
        # There are different branches depending on the number of samples found:
        results = filter_by_cpt(seed_data, cpt_code)
        selected_samples = select_random_rows(results)

        if selected_samples.shape[0] >= 2:
            # If two or more samples are found, select two for context.
            codes_1 = selected_samples[0]["all_codes"]
            approx_synonyms_1 = selected_samples[0]["approx_synonyms"]
            medical_record_1 = selected_samples[0]["text"]

            codes_2 = selected_samples[1]["all_codes"]
            approx_synonyms_2 = selected_samples[1]["approx_synonyms"]
            medical_record_2 = selected_samples[1]["text"]

        elif selected_samples.shape[0] == 1:
            # - If only one sample is found, use it along with another random sample from the seed data.
            codes_1 = selected_samples[0]["all_codes"]
            approx_synonyms_1 = selected_samples[0]["approx_synonyms"]
            medical_record_1 = selected_samples[0]["text"]

            rand_val_1 = generate_random_int(0, seed_data.shape[0], 0)
            example_2 = seed_data[rand_val_1]
            medical_record_2 = example_2["text"]
            approx_synonyms_2 = filter_by_user_codes(codebank, codes_2)["approx_synonyms"]

        elif selected_samples.shape[0] == 0:
            # If no samples are found, select two random samples from the seed data as fallback.
            print("No corresponding samples were found for the given CPT code(s). Using randomly selected samples instead.")
            rand_val_1 = generate_random_int(0, seed_data.shape[0], 0)
            rand_val_2 = generate_random_int(0, seed_data.shape[0], rand_val_1)

            example_1 = seed_data[rand_val_1]
            example_2 = seed_data[rand_val_2]

            codes_1 = example_1["all_codes"]
            codes_2 = example_2["all_codes"]

            medical_record_1 = example_1["text"]
            medical_record_2 = example_2["text"]

            approx_synonyms_1 = filter_by_user_codes(codebank, codes_1)["approx_synonyms"]
            approx_synonyms_2 = filter_by_user_codes(codebank, codes_2)["approx_synonyms"]

    prompt = f"""###Objective
  Generate a clinical note from the given ICD-10-CM codes and approximate synonyms. Your response should
  follow the formatting and general layout of the example medical notes. You NEVER include
  ICD-10-CM codes in the clinical note. You NEVER use a narrative style. The note should contain
  fragmented information, making it harder to follow a continuous narrative.

  ## EXAMPLE
  **ICD-10-CM Codes**: {codes_1}

  **Approximate Synonyms**: {approx_synonyms_1}

  **Clinical Note**: {medical_record_1}

  **ICD-10-CM Codes**: {codes_2}

  **Approximate Synonyms**: {approx_synonyms_2}

  **Clinical Note**: {medical_record_2}

  **ICD-10-CM Codes**: {user_codes}

  **Approximate Synonyms**: {approx_synonyms}

  **Clinical Note**:"""

    try:
        completion_message = get_chat_completion(prompt, model=model)
        return {"completion": completion_message}

    except RetryError:
        print("Whoops! The selected medical records are too large for 'gpt-3.5-turbo-1106'. Retrying with `gpt-4-0125-preview' instead.")
        completion_message = get_chat_completion(prompt, model="gpt-4-0125-preview")
        return {"completion": completion_message}
