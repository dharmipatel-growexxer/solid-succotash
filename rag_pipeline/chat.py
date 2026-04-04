#!/usr/bin/env python3
"""
Interactive CLI chat interface for the Government Schemes RAG chatbot.
"""
import argparse
import sys
from typing import Optional

from chain import RAGChain


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_colored(text: str, color: str = "", bold: bool = False):
    """Print colored text."""
    prefix = ""
    if bold:
        prefix += Colors.BOLD
    if color:
        prefix += color
    suffix = Colors.ENDC if (color or bold) else ""
    print(f"{prefix}{text}{suffix}")


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🇮🇳  सारथी (Sarathi) - Government Schemes Assistant  🇮🇳      ║
║                                                               ║
║   Find government schemes you're eligible for!                ║
║   Just tell me about yourself and what you're looking for.    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print_colored(banner, Colors.CYAN, bold=True)
    
    print_colored("\nCommands:", Colors.YELLOW)
    print("  /help     - Show this help message")
    print("  /profile  - Show your detected profile")
    print("  /reset    - Start a new conversation")
    print("  /debug    - Toggle debug mode (show retrieval info)")
    print("  /quit     - Exit the chatbot")
    print()
    print_colored("Example questions:", Colors.YELLOW)
    print("  • I'm a 30 year old farmer in Gujarat. What schemes can help me?")
    print("  • Are there any scholarships for SC students in Maharashtra?")
    print("  • What housing schemes are available for poor families?")
    print()


def print_debug_info(docs: list, profile: dict):
    """Print debug information about retrieval."""
    print_colored("\n--- Debug Info ---", Colors.DIM)
    print_colored(f"Extracted profile signals:", Colors.DIM)
    for key, value in profile.items():
        if value:
            print_colored(f"  {key}: {value}", Colors.DIM)
    
    print_colored(f"\nRetrieved {len(docs)} documents:", Colors.DIM)
    for i, doc in enumerate(docs[:5]):  # Show top 5
        meta = doc.get("metadata", {})
        score = doc.get("score", 0)
        print_colored(
            f"  [{i+1}] {meta.get('scheme_name', 'Unknown')[:50]} "
            f"({meta.get('chunk_type', 'N/A')}) - score: {score:.3f}",
            Colors.DIM
        )
    print_colored("--- End Debug ---\n", Colors.DIM)


def chat_loop(chain: RAGChain, debug: bool = False):
    """Main chat loop."""
    debug_mode = debug
    
    while True:
        try:
            # Get user input
            print_colored("\nYou: ", Colors.GREEN, bold=True)
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command in ["/quit", "/exit", "/q"]:
                    print_colored("\nThank you for using Sarathi! Goodbye! 🙏", Colors.CYAN)
                    break
                
                elif command in ["/help", "/h", "/?"] :
                    print_banner()
                    continue
                
                elif command in ["/reset", "/clear", "/new"]:
                    chain.reset()
                    print_colored("✓ Conversation reset. Let's start fresh!", Colors.GREEN)
                    continue
                
                elif command in ["/profile", "/p"]:
                    profile = chain.get_user_profile()
                    print_colored("\n📋 Your Profile (detected from conversation):", Colors.YELLOW)
                    has_info = False
                    for key, value in profile.items():
                        if value:
                            has_info = True
                            display_key = key.replace("_", " ").title()
                            print(f"  • {display_key}: {value}")
                    if not has_info:
                        print("  No profile information detected yet.")
                        print("  Tell me about yourself (state, age, occupation, etc.)")
                    continue
                
                elif command in ["/debug", "/d"]:
                    debug_mode = not debug_mode
                    status = "enabled" if debug_mode else "disabled"
                    print_colored(f"✓ Debug mode {status}", Colors.YELLOW)
                    continue
                
                elif command == "/history":
                    history = chain.get_conversation_history()
                    if not history:
                        print_colored("No conversation history yet.", Colors.DIM)
                    else:
                        print_colored(f"\n📜 Conversation History ({len(history)} turns):", Colors.YELLOW)
                        for i, turn in enumerate(history):
                            print(f"\n[{i+1}] You: {turn['user'][:100]}...")
                            print(f"    Sarathi: {turn['assistant'][:100]}...")
                    continue
                
                else:
                    print_colored(f"Unknown command: {command}. Type /help for available commands.", Colors.RED)
                    continue
            
            # Process query through RAG chain
            print_colored("\nSarathi: ", Colors.BLUE, bold=True)
            print_colored("Thinking...", Colors.DIM)
            
            try:
                response, docs, profile = chain.query(user_input)
                
                # Clear "Thinking..." line and print response
                print("\033[A\033[K", end="")  # Move up and clear line
                print(response)
                
                # Show debug info if enabled
                if debug_mode:
                    print_debug_info(docs, profile.to_dict())
                
            except Exception as e:
                print("\033[A\033[K", end="")  # Clear "Thinking..."
                print_colored(f"Error: {str(e)}", Colors.RED)
                if debug_mode:
                    import traceback
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print_colored("\n\nInterrupted. Type /quit to exit.", Colors.YELLOW)
            continue
        
        except EOFError:
            print_colored("\n\nGoodbye! 🙏", Colors.CYAN)
            break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sarathi - Government Schemes Chatbot"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode (show retrieval info)"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the welcome banner"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode (non-interactive)"
    )
    
    args = parser.parse_args()
    
    # Initialize chain
    try:
        print_colored("Initializing Sarathi...", Colors.DIM)
        chain = RAGChain()
        print("\033[A\033[K", end="")  # Clear initialization message
    except Exception as e:
        print_colored(f"Failed to initialize: {e}", Colors.RED)
        print("\nMake sure you have:")
        print("  1. Set GOOGLE_API_KEY in .env file")
        print("  2. Set PINECONE_API_KEY in .env file")
        print("  3. Installed required packages: pip install google-generativeai pinecone")
        sys.exit(1)
    
    # Single query mode
    if args.query:
        response, docs, profile = chain.query(args.query)
        print(response)
        if args.debug:
            print_debug_info(docs, profile.to_dict())
        return
    
    # Interactive mode
    if not args.no_banner:
        print_banner()
    
    chat_loop(chain, debug=args.debug)


if __name__ == "__main__":
    main()
