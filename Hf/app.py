import os
import threading
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    SearchInformationTool,
    SimpleTextBrowser,
)
from smolagents import LiteLLMModel, ToolCallingAgent

# Load environment variables and login
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

# Browser configuration
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
append_answer_lock = threading.Lock()

def process_question(question, model_id="o1"):
    text_limit = 100000
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    
    # Initialize model and tools
    model = LiteLLMModel(
        model_id,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        reasoning_effort="high",
    )
    document_inspection_tool = TextInspectorTool(model, text_limit)
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    
    # Initialize tools
    tools = [SearchInformationTool(browser)]
    
    # Create agent
    agent = ToolCallingAgent(
        model=model,
        tools=tools,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!""",
        provide_run_summary=True,
    )
    
    # Get response
    response = agent.run(question)
    return response

# Create Gradio interface
def answer_question(question, model_id):
    try:
        response = process_question(question, model_id)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Define CSS
css = """
.gradio-container {
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Helvetica Neue", "Ubuntu", sans-serif;
}
"""

# Define markdown content
title = "# open Deep Research - free the AI agents! 🚀"
description = """
Built with [smolagents](https://github.com/huggingface/smolagents)

OpenAI just published [Deep Research](https://openai.com/index/introducing-deep-research/), a very nice assistant that can perform deep searches on the web to answer user questions.

However, their agent has a huge downside: it's not open. So we've started a 24-hour rush to replicate and open-source it. Our resulting [open-Deep-Research agent](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research) took the #1 rank of any open submission on the GAIA leaderboard! ✨

You can try a simplified version below. 👇
"""

# Define the Gradio interface with updated styling
with gr.Blocks(css=css) as iface:
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Row():
        output = gr.Markdown(label="Answer")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask me anything that requires web research...",
                lines=3
            )
            model_dropdown = gr.Dropdown(
                choices=["o1", "gpt-4", "gpt-3.5-turbo"],
                value="o1",
                label="Model"
            )
            submit_btn = gr.Button("Research!", variant="primary")
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, model_dropdown],
        outputs=output
    )
    
    gr.Markdown("Built with [Gradio](https://gradio.app)")

if __name__ == "__main__":
    iface.launch(share=True)