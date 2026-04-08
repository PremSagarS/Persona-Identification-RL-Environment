import os
import openai

from dotenv import load_dotenv

try:
    load_dotenv()
except:
    pass

class PersistentLLMHelper:
    def __init__(self, system_prompt):
        # Fetch configurations with fallbacks
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("LLM_MODEL_NAME")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # This list stores the conversation history
        self.history = [
            {"role": "system", "content": system_prompt}
        ]

    def prompt(self, msg: str) -> str:
        """
        Sends a message while maintaining conversation context.
        """
        # 1. Add the user's new message to the history
        self.history.append({"role": "user", "content": msg})

        try:
            # 2. Send the ENTIRE history to the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history
            )
            
            reply = response.choices[0].message.content

            # 3. Add the LLM's reply to history so it remembers what it said
            self.history.append({"role": "assistant", "content": reply})
            
            return reply

        except Exception as e:
            return f"An error occurred: {e}"

    def clear_history(self):
        """Resets the conversation context."""
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]

# --- Example of Persistence ---
if __name__ == "__main__":
    chat = PersistentLLMHelper()
    print(chat.prompt("My name is Gemini.")) 
    print(chat.prompt("What is my name?")) # The LLM will remember "Gemini"