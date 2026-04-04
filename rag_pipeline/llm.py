"""
LLM wrapper for Groq.
Groq offers free API access with fast inference on open-source models.
"""
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from groq import RateLimitError, APIError

from config import LLM_MODEL, LLM_TEMPERATURE

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds


def get_client() -> Groq:
    """Get configured Groq client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Get a free API key from: "
            "https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


class GroqLLM:
    """
    Wrapper for Groq LLM.
    
    Supports:
    - Single message generation
    - Multi-turn conversations
    - System instructions
    
    Available models (free tier):
    - llama-3.3-70b-versatile (best quality)
    - llama-3.1-8b-instant (fastest)
    - mixtral-8x7b-32768 (good for long context)
    - gemma2-9b-it (Google's model)
    """
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        system_instruction: Optional[str] = None,
    ):
        self.client = get_client()
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction
        self.max_tokens = 2048
        
        # Chat history (for multi-turn)
        self._chat_history: List[Dict[str, str]] = []
        
        # Initialize with system message if provided
        if system_instruction:
            self._system_message = {"role": "system", "content": system_instruction}
        else:
            self._system_message = None
    
    def _get_messages(self, include_history: bool = True) -> List[Dict[str, str]]:
        """Build messages list with system prompt and history."""
        messages = []
        if self._system_message:
            messages.append(self._system_message)
        if include_history:
            messages.extend(self._chat_history)
        return messages
    
    def _call_with_retry(self, messages: List[Dict[str, str]], max_retries: int = MAX_RETRIES) -> str:
        """Call API with retry logic for rate limits."""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    print(f"Rate limited. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise
            except APIError as e:
                last_error = e
                if "rate" in str(e).lower() and attempt < max_retries - 1:
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    print(f"API error. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise
        raise last_error

    def generate(self, prompt: str) -> str:
        """
        Generate a single response (stateless).
        
        Args:
            prompt: The user prompt
            
        Returns:
            Generated text response
        """
        messages = []
        if self._system_message:
            messages.append(self._system_message)
        messages.append({"role": "user", "content": prompt})
        
        return self._call_with_retry(messages)
    
    def start_chat(self, history: Optional[List[Dict[str, str]]] = None):
        """
        Start a new chat session.
        
        Args:
            history: Optional conversation history in format:
                     [{"role": "user", "content": "text"}, 
                      {"role": "assistant", "content": "text"}]
        """
        self._chat_history = []
        if history:
            for item in history:
                role = item.get("role", "user")
                # Normalize role names (Gemini uses "model", Groq uses "assistant")
                if role == "model":
                    role = "assistant"
                content = item.get("content") or (item.get("parts", [""])[0] if "parts" in item else "")
                self._chat_history.append({"role": role, "content": content})
    
    def chat(self, message: str) -> str:
        """
        Send a message in the current chat session.
        
        Args:
            message: User message
            
        Returns:
            Model response
        """
        # Add user message to history
        self._chat_history.append({"role": "user", "content": message})
        
        # Build full messages list
        messages = self._get_messages(include_history=True)
        
        # Generate response with retry logic
        try:
            assistant_text = self._call_with_retry(messages)
        except Exception as e:
            # Remove the user message if generation fails
            self._chat_history.pop()
            raise
        
        # Add assistant response to history
        self._chat_history.append({"role": "assistant", "content": assistant_text})
        
        return assistant_text
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the current chat history."""
        return [
            {"role": msg["role"], "parts": [msg["content"]]}
            for msg in self._chat_history
        ]
    
    def clear_chat(self):
        """Clear the chat session."""
        self._chat_history = []


# === Backward compatibility aliases ===
GeminiLLM = GroqLLM  # Alias for code that imports GeminiLLM


# === Singleton Instance ===
_llm = None


def get_llm(system_instruction: Optional[str] = None) -> GroqLLM:
    """Get or create singleton LLM instance."""
    global _llm
    if _llm is None or (_llm.system_instruction != system_instruction):
        _llm = GroqLLM(system_instruction=system_instruction)
    return _llm


if __name__ == "__main__":
    # Test the LLM
    print("Testing Groq LLM...")
    print(f"Model: {LLM_MODEL}")
    
    llm = GroqLLM(
        system_instruction="You are a helpful assistant specializing in Indian government schemes. Keep responses concise."
    )
    
    # Test single generation
    print("\n=== Single Generation Test ===")
    response = llm.generate("What is PM-KISAN scheme? Answer in 2 sentences.")
    print(f"Response: {response}")
    
    # Test multi-turn chat
    print("\n=== Multi-turn Chat Test ===")
    llm.start_chat()
    
    messages = [
        "I'm a farmer in Gujarat. What schemes might help me?",
        "Tell me more about the first scheme you mentioned.",
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = llm.chat(msg)
        print(f"Assistant: {response[:500]}...")
    
    print("\n=== Chat History ===")
    history = llm.get_chat_history()
    print(f"Total turns: {len(history)}")
    print("\nTest complete!")
