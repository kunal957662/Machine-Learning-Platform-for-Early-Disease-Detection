import os
import json
import mimetypes
import base64

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

try:
    import streamlit as st
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=api_key)


def extract_text_from_image(image_path):
    """
    Extract medical report information from an image
    using Groq Vision.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Detect image type automatically
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise ValueError("Unsupported image format. Use JPG, PNG, or WEBP.")

    # Read image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Convert image to base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    image_data_url = f"data:{mime_type};base64,{image_base64}"

    prompt = """
You are a medical report OCR assistant.

Read the uploaded medical report carefully.

Extract ONLY information that is clearly visible.

DO NOT guess missing values.

Return ONLY valid JSON in this format:

{
    "patient_name": "",
    "age": null,
    "gender": "",
    "medical_values": {},
    "other_information": ""
}

Rules:
1. Extract only clearly readable information.
2. Never guess or estimate values.
3. Keep numerical values exactly as shown.
4. Put medical measurements inside "medical_values".
5. If a value is missing or unreadable, use null or "".
"""

    try:

        response = client.chat.completions.create(
           model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_object"
            }
        )

        result = response.choices[0].message.content

        return json.loads(result)

    except json.JSONDecodeError:

        return {
            "raw_text": result
        }

    except Exception as e:

        return {
            "error": str(e)
        }


if __name__ == "__main__":
    print("Groq Vision OCR reader loaded successfully.")