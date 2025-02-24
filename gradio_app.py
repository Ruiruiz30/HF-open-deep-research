from dotenv import load_dotenv
import os
from smolagents import CodeAgent, GradioUI, LiteLLMModel, ToolCallingAgent
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

# 从run.py导入必要的配置
from run import AUTHORIZED_IMPORTS, BROWSER_CONFIG

# 设置环境变量
load_dotenv(override=True)
if os.getenv("HF_TOKEN"):
    from huggingface_hub import login
    login(os.getenv("HF_TOKEN"))

def create_agent():
    text_limit = 100000
    model = LiteLLMModel(
        "o1",  # 使用默认model_id
        custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
        max_completion_tokens=8192,
        reasoning_effort="high",
    )

    document_inspection_tool = TextInspectorTool(model, text_limit)
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    # 初始化Web工具
    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    # 创建web浏览agent
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    # 创建主agent
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, document_inspection_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent

def main():
    # 创建agent
    agent = create_agent()
    # 创建并启动Gradio界面
    ui = GradioUI(agent, file_upload_folder="./data")
    # ui.launch(share=True)  # share=True允许通过公共URL访问
    GradioUI(agent).launch()
if __name__ == "__main__":
    main()
