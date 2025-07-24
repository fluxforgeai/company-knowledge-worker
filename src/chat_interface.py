"""
Gradio chat interface for Company Knowledge Worker
"""

import logging
from typing import List, Tuple, Optional

import gradio as gr

from .rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class ChatInterface:
    """Manages the Gradio chat interface"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.interface = None
        
    def chat_function(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Chat function for Gradio interface"""
        try:
            result = self.rag_pipeline.ask_question(message)
            return result["answer"]
        except Exception as e:
            logger.error(f"Chat function error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio chat interface with persistent examples"""
        try:
            with gr.Blocks(
                title="🏢 Artiligence Knowledge Worker",
                theme=gr.themes.Soft(),
                css=self._get_custom_css_with_examples()
            ) as interface:
                
                # Header
                gr.Markdown("# 🏢 Artiligence Knowledge Worker")
                gr.Markdown(self._get_description())
                
                # Persistent example questions at the top
                gr.Markdown("## 💡 Quick Questions (Click to use):")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        example_buttons = []
                        examples = self._get_examples()
                        
                        # Create buttons for first half of examples
                        for i in range(0, len(examples), 2):
                            if i < len(examples):
                                btn = gr.Button(examples[i], size="sm", variant="secondary")
                                example_buttons.append((btn, examples[i]))
                    
                    with gr.Column(scale=1):
                        # Create buttons for second half of examples
                        for i in range(1, len(examples), 2):
                            if i < len(examples):
                                btn = gr.Button(examples[i], size="sm", variant="secondary")
                                example_buttons.append((btn, examples[i]))
                
                gr.Markdown("---")
                
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=False,
                    container=True,
                    bubble_full_width=False
                )
                
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask me anything about Artiligence...",
                    container=False,
                    scale=7
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                
                # Chat functionality
                def respond(message, history):
                    if not message.strip():
                        return history, ""
                    
                    try:
                        result = self.rag_pipeline.ask_question(message)
                        response = result["answer"]
                        history.append([message, response])
                        return history, ""
                    except Exception as e:
                        logger.error(f"Chat function error: {e}")
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        history.append([message, error_msg])
                        return history, ""
                
                def clear_conversation():
                    return [], ""
                
                # Event handlers
                msg.submit(respond, [msg, chatbot], [chatbot, msg])
                submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
                clear_btn.click(clear_conversation, None, [chatbot, msg])
                
                # Connect example buttons with proper closure handling
                def create_example_handler(text):
                    def handler():
                        return text
                    return handler
                
                for btn, example_text in example_buttons:
                    btn.click(create_example_handler(example_text), None, msg, show_progress=False)
            
            self.interface = interface
            logger.info("Enhanced Gradio chat interface with persistent examples created successfully")
            return interface
            
        except Exception as e:
            logger.error(f"Error creating enhanced Gradio interface: {e}")
            raise
    
    def _get_description(self) -> str:
        """Get the interface description"""
        return """
        Ask me anything about Artiligence company documents, projects, invoices, contracts, and more! 
        
        **Enhanced with comprehensive project knowledge including:**
        - SQL Server upgrades and database infrastructure
        - Precision agriculture and asset management systems  
        - Mining analytics and fleet management (SFMS)
        - Database migration projects
        - Advanced driver assistance systems (ADAS)
        
        **Document types covered:**
        - Company documents and contracts
        - Project documentation and specifications
        - Financial records and invoices
        - Technical documentation and scripts
        """
    
    def _get_examples(self) -> List[str]:
        """Get example questions for the interface"""
        return [
            "What projects is Artiligence working on?",
            "Tell me about the SQL Server upgrade project",
            "What is the Precision Agriculture Asset Management project?",
            "Describe the SFMS Mining Analytics project", 
            "What database migration work is being done?",
            "Tell me about the company's invoices and financial information",
            "What contracts does Artiligence have?",
            "Summarize the company's business activities",
            "What technical documentation is available?",
            "Tell me about the Advanced Driver Assistance System project"
        ]
    
    def _get_custom_css_with_examples(self) -> str:
        """Get custom CSS for the enhanced interface with persistent examples"""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            font-size: 16px !important;
        }
        /* Style for example buttons */
        button[variant="secondary"] {
            margin: 2px !important;
            padding: 8px 12px !important;
            font-size: 13px !important;
            line-height: 1.2 !important;
            text-align: left !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            min-height: 45px !important;
            max-height: 60px !important;
        }
        /* Ensure buttons expand to fill column width */
        .gr-button {
            width: 100% !important;
        }
        /* Style the header sections */
        h1, h2 {
            margin-bottom: 10px !important;
        }
        /* Chat container styling */
        .chatbot {
            border-radius: 8px !important;
        }
        """
    
    def launch(self, 
               server_name: str = "127.0.0.1", 
               server_port: Optional[int] = None,
               share: bool = False,
               inbrowser: bool = True) -> None:
        """Launch the enhanced Gradio interface"""
        if not self.interface:
            raise ValueError("Interface not created. Call create_interface() first.")
        
        try:
            # Import config here to avoid circular imports
            from .config import config
            
            launch_kwargs = {
                "server_name": server_name,
                "share": share,
                "inbrowser": inbrowser
            }
            
            # Handle port selection more intelligently
            if server_port is not None:
                launch_kwargs["server_port"] = server_port
            # If no port specified, let Gradio find an available one
            # Don't set server_port at all to enable automatic port selection
            
            logger.info(f"Launching enhanced Gradio interface with kwargs: {launch_kwargs}")
            self.interface.launch(**launch_kwargs)
            
        except Exception as e:
            logger.error(f"Error launching enhanced Gradio interface: {e}")
            raise
    
    def get_interface_info(self) -> dict:
        """Get information about the interface"""
        return {
            "interface_created": self.interface is not None,
            "interface_type": "Enhanced with persistent examples",
            "rag_pipeline_ready": self.rag_pipeline.get_pipeline_status()["overall_ready"],
            "examples_count": len(self._get_examples())
        }

class SimpleChatInterface:
    """Simple command-line chat interface as fallback"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def run(self):
        """Run the simple command-line chat interface"""
        print("\\n" + "="*60)
        print("🏢 ARTILIGENCE KNOWLEDGE WORKER - SIMPLE CHAT")
        print("="*60)
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type 'help' for available commands")
        print("Ask me anything about your company documents!")
        print("="*60)
        
        while True:
            try:
                question = input("\\n❓ Your question: ")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if question.lower() == 'status':
                    self._show_status()
                    continue
                
                if question.lower() == 'clear':
                    self.rag_pipeline.clear_conversation_history()
                    print("🧹 Conversation history cleared!")
                    continue
                    
                if not question.strip():
                    continue
                    
                print("🤔 Thinking...")
                
                result = self.rag_pipeline.ask_question(question)
                
                if result["success"]:
                    print(f"\\n💬 Answer: {result['answer']}")
                else:
                    print(f"\\n❌ Error: {result.get('error', 'Unknown error')}")
                
                print("\\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Unexpected error: {e}")
                logger.error(f"Simple chat error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\\n📚 Available commands:")
        print("  • help    - Show this help message")
        print("  • status  - Show system status")
        print("  • clear   - Clear conversation history")
        print("  • quit    - Exit the application")
        print("\\n💡 You can ask questions about:")
        print("  • Artiligence projects and documentation")
        print("  • Company contracts and invoices")
        print("  • Technical specifications and procedures")
        print("  • Financial information and records")
    
    def _show_status(self):
        """Show system status"""
        status = self.rag_pipeline.get_pipeline_status()
        vector_stats = status.get("vector_store_stats", {})
        
        print("\\n📊 System Status:")
        print(f"  • RAG Pipeline Ready: {'✅' if status['overall_ready'] else '❌'}")
        print(f"  • Vector Store: {'✅' if status['vector_store_available'] else '❌'}")
        print(f"  • LLM Initialized: {'✅' if status['llm_initialized'] else '❌'}")
        print(f"  • Memory Initialized: {'✅' if status['memory_initialized'] else '❌'}")
        
        if vector_stats:
            print(f"  • Total Documents: {vector_stats.get('total_documents', 'Unknown')}")
            print(f"  • Embedding Dimensions: {vector_stats.get('embedding_dimensions', 'Unknown')}")
            
            doc_breakdown = vector_stats.get('doc_type_breakdown', {})
            if doc_breakdown:
                print("  • Document Types:")
                for doc_type, count in doc_breakdown.items():
                    print(f"    - {doc_type}: {count}")