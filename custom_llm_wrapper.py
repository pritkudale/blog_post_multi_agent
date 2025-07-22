import os
from langchain_openai import ChatOpenAI
from langchain.schema.output import ChatGeneration

# A simple list to store the model names from each call
# In a more complex app, you might use logging or a more robust storage mechanism
MODEL_USAGE_LOG = []

class ModelInterceptLLM(ChatOpenAI):
    """
    A custom LLM wrapper to intercept the model name from the API response.
    """
    def _generate(self, *args, **kwargs):
        """
        Overrides the _generate method to intercept the response.
        """
        # Call the original method to get the actual API response
        response = super()._generate(*args, **kwargs)

        # The raw response data is in the `response_metadata` of the first generation
        if response.generations and isinstance(response.generations[0], ChatGeneration):
            response_metadata = response.generations[0].response_metadata
            if "model" in response_metadata:
                model_name = response_metadata["model"]
                
                # Log the model name so we can see it was captured
                print(f"--- Captured model from API call: {model_name} ---")
                
                # Store it for later access if needed
                MODEL_USAGE_LOG.append(model_name)

        return response