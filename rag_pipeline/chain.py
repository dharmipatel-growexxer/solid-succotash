"""
RAG Chain - Combines retrieval with LLM generation.
Includes conversation memory and context management.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import MEMORY_WINDOW_SIZE, RETRIEVAL_K
from llm import GeminiLLM
from retriever import (
    SchemeRetriever,
    UserProfile,
    extract_user_profile,
    format_retrieved_docs,
)


# === System Prompt ===
SYSTEM_PROMPT = """You are **Sarathi** (सारथी), an AI assistant helping Indian citizens discover and understand government welfare schemes they're eligible for.

## Your Role
You help users find relevant central and state government schemes based on their profile (location, age, occupation, gender, caste, income, etc.). You provide accurate, helpful information about:
- Scheme eligibility criteria
- Benefits and entitlements  
- Application process and required documents
- Where and how to apply

## Guidelines

### Information Accuracy
- ONLY provide information from the retrieved scheme documents
- If information is not in the retrieved context, say "I don't have specific information about that in my current knowledge base"
- Never make up scheme details, eligibility criteria, or benefits
- Always mention the scheme name clearly when discussing it

### Communication Style
- Be warm, helpful, and patient - many users may be first-time scheme seekers
- Use simple language; avoid bureaucratic jargon
- When relevant, mention if you need more information to give better recommendations
- Structure responses clearly with bullet points or numbered lists for complex information

### Profile Understanding
- Pay attention to user details: state, age, gender, occupation, caste category (SC/ST/OBC), income level, BPL status
- Proactively ask clarifying questions if key profile information is missing
- Remember context from earlier in the conversation

### Response Format
When discussing schemes, include:
1. **Scheme Name** - Full official name
2. **Key Benefits** - What the user gets
3. **Eligibility** - Who can apply (highlight if user qualifies)
4. **How to Apply** - Brief process overview
5. **Documents Needed** - Key documents required

### Important Notes
- Encourage users to verify current information on official portals as schemes may update
- Be inclusive and respectful of all backgrounds
- If asked about schemes outside India, politely clarify your focus is on Indian government schemes

## Context
You will receive relevant scheme information in the <context> section. Use this to answer user queries accurately."""


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    user_message: str
    assistant_response: str
    retrieved_docs: List[Dict] = field(default_factory=list)
    user_profile: Optional[UserProfile] = None


@dataclass
class ConversationMemory:
    """
    Manages conversation history with a sliding window.
    """
    max_turns: int = MEMORY_WINDOW_SIZE
    turns: List[ConversationTurn] = field(default_factory=list)
    cumulative_profile: UserProfile = field(default_factory=UserProfile)
    
    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        retrieved_docs: List[Dict] = None,
        user_profile: Optional[UserProfile] = None,
    ):
        """Add a conversation turn."""
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            retrieved_docs=retrieved_docs or [],
            user_profile=user_profile,
        )
        self.turns.append(turn)
        
        # Trim to max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        
        # Update cumulative profile
        if user_profile:
            self._merge_profile(user_profile)
    
    def _merge_profile(self, new_profile: UserProfile):
        """Merge new profile info into cumulative profile."""
        if new_profile.state:
            self.cumulative_profile.state = new_profile.state
            self.cumulative_profile.location_type = new_profile.location_type
        if new_profile.age:
            self.cumulative_profile.age = new_profile.age
        if new_profile.gender:
            self.cumulative_profile.gender = new_profile.gender
        if new_profile.occupation:
            self.cumulative_profile.occupation = new_profile.occupation
        if new_profile.income:
            self.cumulative_profile.income = new_profile.income
        if new_profile.caste_category:
            self.cumulative_profile.caste_category = new_profile.caste_category
        if new_profile.is_bpl:
            self.cumulative_profile.is_bpl = True
        if new_profile.is_disabled:
            self.cumulative_profile.is_disabled = True
        
        # Merge category hints (keep unique)
        for cat in new_profile.category_hints:
            if cat not in self.cumulative_profile.category_hints:
                self.cumulative_profile.category_hints.append(cat)
    
    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """
        Format history for Gemini chat.
        Returns list of {"role": "user"/"model", "parts": ["text"]}
        """
        history = []
        for turn in self.turns:
            history.append({"role": "user", "parts": [turn.user_message]})
            history.append({"role": "model", "parts": [turn.assistant_response]})
        return history
    
    def get_context_summary(self) -> str:
        """Get a summary of what we know about the user."""
        profile = self.cumulative_profile
        parts = []
        
        if profile.state:
            parts.append(f"Location: {profile.state}")
        if profile.age:
            parts.append(f"Age: {profile.age}")
        if profile.gender:
            parts.append(f"Gender: {profile.gender}")
        if profile.occupation:
            parts.append(f"Occupation: {profile.occupation}")
        if profile.caste_category:
            parts.append(f"Category: {profile.caste_category.upper()}")
        if profile.income:
            parts.append(f"Income: ₹{profile.income:,}")
        if profile.is_bpl:
            parts.append("BPL: Yes")
        if profile.is_disabled:
            parts.append("Disability: Yes")
        
        if parts:
            return "Known user info: " + ", ".join(parts)
        return ""
    
    def clear(self):
        """Clear conversation history."""
        self.turns = []
        self.cumulative_profile = UserProfile()


class RAGChain:
    """
    RAG Chain combining retrieval with Gemini LLM.
    
    Flow:
    1. Extract user profile from query
    2. Retrieve relevant scheme chunks
    3. Build prompt with context
    4. Generate response with Gemini
    5. Store in memory
    """
    
    def __init__(
        self,
        retrieval_k: int = RETRIEVAL_K,
        use_mmr: bool = True,
    ):
        self.retriever = SchemeRetriever(k=retrieval_k)
        self.llm = GeminiLLM(system_instruction=SYSTEM_PROMPT)
        self.memory = ConversationMemory()
        self.use_mmr = use_mmr
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        if not retrieved_docs:
            return "No specific schemes found matching the query."
        
        context_parts = []
        seen_schemes = set()
        
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            scheme_name = meta.get("scheme_name", "Unknown")
            chunk_type = meta.get("chunk_type", "info")
            text = meta.get("text", "")
            location = meta.get("location_name", "India")
            
            # Track unique schemes
            seen_schemes.add(scheme_name)
            
            context_parts.append(
                f"**{scheme_name}** ({location}) - {chunk_type.replace('_', ' ').title()}:\n{text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        user_query: str,
        context: str,
        user_profile: UserProfile,
    ) -> str:
        """Build the full prompt for the LLM."""
        # Profile summary
        profile_summary = self.memory.get_context_summary()
        
        prompt_parts = []
        
        # Add context
        prompt_parts.append("<context>")
        prompt_parts.append(context)
        prompt_parts.append("</context>")
        
        # Add profile context if available
        if profile_summary:
            prompt_parts.append(f"\n<user_profile>\n{profile_summary}\n</user_profile>")
        
        # Add current query info
        current_profile = user_profile.to_dict()
        current_profile_str = ", ".join(
            f"{k}: {v}" for k, v in current_profile.items() if v
        )
        if current_profile_str:
            prompt_parts.append(f"\n<current_query_signals>\n{current_profile_str}\n</current_query_signals>")
        
        # Add user query
        prompt_parts.append(f"\n<user_query>\n{user_query}\n</user_query>")
        
        return "\n".join(prompt_parts)
    
    def query(
        self,
        user_query: str,
        k: Optional[int] = None,
    ) -> Tuple[str, List[Dict], UserProfile]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            user_query: The user's question
            k: Optional override for number of docs to retrieve
            
        Returns:
            Tuple of (response, retrieved_docs, extracted_profile)
        """
        # Extract profile from current query
        current_profile = extract_user_profile(user_query)
        
        # Use cumulative profile for better filtering
        # but prioritize current query signals
        retrieval_profile = UserProfile(
            state=current_profile.state or self.memory.cumulative_profile.state,
            location_type=current_profile.location_type or self.memory.cumulative_profile.location_type,
            category_hints=current_profile.category_hints or self.memory.cumulative_profile.category_hints,
            gender=current_profile.gender or self.memory.cumulative_profile.gender,
            occupation=current_profile.occupation or self.memory.cumulative_profile.occupation,
            age=current_profile.age or self.memory.cumulative_profile.age,
            income=current_profile.income or self.memory.cumulative_profile.income,
            caste_category=current_profile.caste_category or self.memory.cumulative_profile.caste_category,
            is_bpl=current_profile.is_bpl or self.memory.cumulative_profile.is_bpl,
            is_disabled=current_profile.is_disabled or self.memory.cumulative_profile.is_disabled,
            raw_query=user_query,
        )
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=user_query,
            profile=retrieval_profile,
            k=k,
            use_mmr=self.use_mmr,
        )
        
        # Build context and prompt
        context = self._build_context(retrieved_docs)
        prompt = self._build_prompt(user_query, context, current_profile)
        
        # Generate response
        # For first message, use generate; for subsequent, use chat with history
        if len(self.memory.turns) == 0:
            self.llm.start_chat()
        
        response = self.llm.chat(prompt)
        
        # Store in memory
        self.memory.add_turn(
            user_message=user_query,
            assistant_response=response,
            retrieved_docs=retrieved_docs,
            user_profile=current_profile,
        )
        
        return response, retrieved_docs, current_profile
    
    def reset(self):
        """Reset conversation state."""
        self.memory.clear()
        self.llm.clear_chat()
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get the cumulative user profile."""
        return self.memory.cumulative_profile.to_dict()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [
            {
                "user": turn.user_message,
                "assistant": turn.assistant_response,
            }
            for turn in self.memory.turns
        ]


# === Singleton Instance ===
_chain = None


def get_chain() -> RAGChain:
    """Get or create singleton RAG chain instance."""
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain


if __name__ == "__main__":
    # Test the RAG chain
    print("=" * 60)
    print("RAG Chain Test")
    print("=" * 60)
    
    chain = RAGChain()
    
    test_queries = [
        "I'm a 25 year old farmer in Maharashtra. What schemes can help me?",
        "Tell me more about any crop insurance schemes",
        "What documents would I need to apply?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print("=" * 60)
        
        response, docs, profile = chain.query(query)
        
        print(f"\nExtracted Profile: {profile.to_dict()}")
        print(f"Retrieved {len(docs)} documents")
        print(f"\nAssistant: {response}")
    
    print(f"\n{'='*60}")
    print("Cumulative User Profile:")
    print(chain.get_user_profile())
