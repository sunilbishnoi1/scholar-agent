from dotenv import load_dotenv

load_dotenv()

import logging
import os
import time

import requests


class GeminiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

    def chat(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        retries = 5
        backoff_factor = 2

        for i in range(retries):
            try:
                resp = requests.post(self.base_url, headers=headers, params=params, json=data, timeout=60)
                resp.raise_for_status()

                result = resp.json()
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    return str(result)

            except requests.exceptions.HTTPError as e:
                # Retry on server errors (5xx) AND rate limit errors (429)
                if (e.response.status_code >= 500 or e.response.status_code == 429) and i < retries - 1:
                    wait_time = backoff_factor ** i
                    logging.warning(f"API Error ({e.response.status_code}): {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to get response from Gemini API after multiple retries: {e}")
                    raise e
            except requests.exceptions.RequestException as e:
                logging.error(f"A network error occurred: {e}")
                if i < retries - 1:
                    wait_time = backoff_factor ** i
                    logging.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e

        return "Error: Could not retrieve a valid response from the model after multiple retries."
