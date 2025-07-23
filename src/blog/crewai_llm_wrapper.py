# blog/crewai_llm_wrapper.py
import os
from typing import Any, Dict, List, Optional, Union
from crewai.llm import LLM as CrewAI_LLM
import requests

MODEL_USAGE_LOG = []

class InterceptedCrewAILLM(CrewAI_LLM):
    """
    Custom CrewAI LLM that intercepts calls to capture model usage
    """
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, 
                 endpoint: Optional[str] = None, temperature: Optional[float] = 0.7, 
                 base_url: Optional[str] = None, **kwargs):
        """
        Initializes the custom LLM.

        Args:
            model: The model name to be used.
            api_key: The API key for authentication.
            endpoint: The API endpoint for requests.
            temperature: The temperature for the model's responses.
            base_url: Alternative parameter name for endpoint (CrewAI compatibility).
            **kwargs: Additional keyword arguments for CrewAI compatibility.
        """
        # Get model from environment if not provided
        if not model:
            model = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
        
        # Handle base_url as an alternative to endpoint
        if base_url and not endpoint:
            endpoint = base_url
        
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        
        # Set endpoint from environment or default
        if not endpoint:
            endpoint = os.getenv("OPENAI_API_BASE", "https://api.dynaroute.vizuara.com/chat/completions")
        
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.api_key = api_key
        self.endpoint = endpoint
        
        print(f"Initialized InterceptedCrewAILLM with model: {model}, endpoint: {endpoint}")
    
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Makes a call to the custom LLM.

        Args:
            messages: The messages to send to the LLM, can be a string or a list of dicts.
            tools: Optional list of tools the LLM can use.

        Returns:
            The LLM's response as a string.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            print("Sending request to API...")
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=120  # Adding a timeout for the request
            )
            response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            
            # Extract the response content
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            model_used = result.get("choices", [{}])[0].get("message", {}).get("model") or result.get("model", self.model)
            
            MODEL_USAGE_LOG.append({
                "model": model_used,
                "response": content
            })
            print(f"Used model: {model_used}")
            return content or "No response content received"
            
        except requests.exceptions.RequestException as e:
            # Handle network-related errors
            error_msg = f"Error making request to API: {e}"
            print(error_msg)
            return error_msg
        except (KeyError, IndexError) as e:
            # Handle unexpected response structure
            error_msg = f"Error parsing response from API: {e}. Response: {response.text if 'response' in locals() else 'No response'}"
            print(error_msg)
            return error_msg
        except Exception as e:
            # Handle any other unexpected errors
            error_msg = f"Unexpected error in LLM call: {e}"
            print(error_msg)
            return error_msg
        except (KeyError, IndexError) as e:
            # Handle unexpected response structure
            return f"Error parsing response from API: {e}. Response: {response.text}"
    
    def supports_stop_words(self) -> bool:
        """
        Returns whether this LLM supports stop words.
        Most modern LLMs support stop words, so we return True.
        """
        return True
    
    def get_num_tokens(self, text: str) -> int:
        """
        Returns the number of tokens for the given text.
        This is a rough estimate - you might want to use a proper tokenizer.
        """
        # Rough estimate: ~4 characters per token for most models
        return len(text) // 4
    
    def get_num_tokens_from_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Returns the number of tokens for the given messages.
        """
        total_text = ""
        for message in messages:
            total_text += message.get("content", "")
        return self.get_num_tokens(total_text)


def print_usage_summary():
    """
    Prints a summary of model usage from the MODEL_USAGE_LOG.
    """
    if not MODEL_USAGE_LOG:
        print("\n=== MODEL USAGE SUMMARY ===")
        print("No model calls were logged.")
        print("===========================\n")
        return
    
    print("\n=== MODEL USAGE SUMMARY ===")
    print(f"Total calls made: {len(MODEL_USAGE_LOG)}")
    
    # Count usage by model
    model_counts = {}
    for entry in MODEL_USAGE_LOG:
        model = entry.get("model", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1
    
    print("\nCalls by model:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} calls")
    
    print("===========================\n")
