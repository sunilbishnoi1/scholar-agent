from dotenv import load_dotenv
load_dotenv()

import requests
import os
import time     
import logging  

class GeminiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"

    def chat(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        retries = 3
        backoff_factor = 2  # The delay will be 2, 4, 8 seconds
        
        for i in range(retries):
            try:
                resp = requests.post(self.base_url, headers=headers, params=params, json=data, timeout=60)
                resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                
                result = resp.json()
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    return str(result)

            except requests.exceptions.HTTPError as e:
                # Only retry on server-side errors (5xx)
                if e.response.status_code >= 500 and i < retries - 1:
                    wait_time = backoff_factor ** i
                    logging.warning(f"Server error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # If it's a client error (4xx) or the last retry, re-raise the exception
                    logging.error(f"Failed to get response from Gemini API: {e}")
                    raise e
            except requests.exceptions.RequestException as e:
                # Handle other network errors like timeouts
                logging.error(f"A network error occurred: {e}")
                if i < retries - 1:
                    wait_time = backoff_factor ** i
                    logging.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
        
        return "Error: Could not retrieve a valid response from the model after multiple retries."