import json
from pathlib import Path
from textwrap import dedent
from typing import List
from phi.assistant import Assistant
from phi.tools import Toolkit
from phi.tools.calculator import Calculator
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.file import FileTools
from phi.knowledge import AssistantKnowledge
from phi.embedder.openai import OpenAIEmbedder
from phi.assistant.duckdb import DuckDbAssistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.utils.log import logger
from phi.vectordb.pgvector import PgVector2
from phi.llm.openai import OpenAIChat
from phi.llm.mistral import Mistral
from phi.llm.ollama import Ollama
from phi.llm.openai.like import OpenAILike
from phi.tools.exa import ExaTools

db_url = "postgresql://postgres:010129@192.168.1.132:5432/postgres"
cwd = Path(__file__).parent.resolve()
scratch_dir = cwd.joinpath("scratch")
if not scratch_dir.exists():
    scratch_dir.mkdir(exist_ok=True, parents=True)

def get_agent(
        llm_id: str = "通义千问",
        data_analyst: bool = False,
        research_assistant: bool = False,
        calculator: bool = True,
        ddg_search: bool = False,
        file_tools: bool = False,
        table_name: str = None,
        debug_mode: bool = False,
)-> Assistant:
    logger.info(f"-*- Creating {llm_id} Agent -*-")
    if  llm_id== 'llama3':
        llm = Ollama(model=llm_id)
    elif llm_id == 'qwen2:7b':
        llm = Ollama(model=llm_id)
    # elif llm_id == 'mistral':
    #     llm = Mistral(model=llm_id)
    # elif  llm_id== 'openai':
    #     llm = OpenAIChat(model=llm_id)
    elif llm_id == "通义千问":
        api_key = "sk-a6c6858ed283411bb45a9fe46afc2953"
        llm = OpenAILike(
            model=llm_id,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:
        raise NotImplementedError(f'Framework {llm_id} not implemented')
    tools: List[Toolkit] = []
    extra_instructions: List[str] = []
    # 计算工具
    if calculator:
        tools.append(
            Calculator(
                add=True,
                subtract=True,
                multiply=True,
                divide=True,
                exponentiate=True,
                factorial=True,
                is_prime=True,
                square_root=True,
            )
        )
    # 网络检索工具
    if ddg_search:
        tools.append(DuckDuckGo(fixed_max_results=3))
    # 文件工具：读取、保存、罗列文件
    if file_tools:
        tools.append(FileTools(base_dir=cwd))
        extra_instructions.append(
            "You can use the `read_file` tool to read a file, `save_file` to save a file, and `list_files` to list files in the working directory."
        )
    team: List[Assistant] = []
    if data_analyst:
        semantic_model = json.dumps({
            "tables": [
                {
                    "name": table_name,
                    "description": "Contains information about coal .",
                    "path": "postgresql://postgres:010129@192.168.1.132:5432/postgres"
                    # Update the path if necessary
                }
            ]
        })

        _data_analyst = DuckDbAssistant(
            name="Data Analyst",
            llm=llm,
            role="Analyze coal data and provide insights",
            semantic_model=semantic_model,
            base_dir=scratch_dir,
        )
        team.append(_data_analyst)
        extra_instructions.append(
            "To answer questions about coal, delegate the task to the `Data Analyst`."
        )

    if research_assistant:
        _research_assistant = Assistant(
            name="Research Assistant",
            role="Write a research report on a given topic",
            llm=OpenAIChat(model=llm_id),
            description="You are a Senior New York Times researcher tasked with writing a cover story research report.",
            instructions=[
                "For a given topic, use the `search_exa` to get the top 10 search results.",
                "Carefully read the results and generate a final - NYT cover story worthy report in the <report_format> provided below.",
                "Make your report engaging, informative, and well-structured.",
                "Remember: you are writing for the New York Times, so the quality of the report is important.",
            ],
            expected_output=dedent(
                """\
            An engaging, informative, and well-structured report in the following format:
            <report_format>
            ## Title

            - **Overview** Brief introduction of the topic.
            - **Importance** Why is this topic significant now?

            ### Section 1
            - **Detail 1**
            - **Detail 2**

            ### Section 2
            - **Detail 1**
            - **Detail 2**

            ## Conclusion
            - **Summary of report:** Recap of the key findings from the report.
            - **Implications:** What these findings mean for the future.

            ## References
            - [Reference 1](Link to Source)
            - [Reference 2](Link to Source)
            </report_format>
            """
            ),
            tools=[ExaTools(num_results=5, text_length_limit=1000)],
            # This setting tells the LLM to format messages in markdown
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_research_assistant)
        extra_instructions.append(
            "To write a research report, delegate the task to the `Research Assistant`. "
            "Return the report in the <report_format> to the user as is, without any additional text like 'here is the report'."
        )
    agent = Assistant(
        name="agent",
        llm=llm,
        description=dedent(
            """\
            You are a powerful AI Agent.
            You have access to a set of tools at your disposal.
            Your goal is to assist the user in the best way possible.\
            """
        ),
        instructions=[
            "When the user sends a message, first **think** and determine if:\n"
            " - You can answer by using a tool available to you\n"
            " - You can search the knowledge base\n"
            " - You can search the internet\n"
            " - You need to ask a clarifying question",
            "If the user asks about coal, first ALWAYS search your knowledge base using the `search_knowledge_base` tool.",
            "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If the user asks to summarize the conversation or if you need to reference your chat history with the user, use the `get_chat_history` tool.",
            "If the user's message is unclear, ask clarifying questions to get more information.",
            "If the user asks you to save, read, and list files, use the 'FileTools' tool."
            "If the user asks you to perform mathematical calculations, use the 'Calculator' tool to do so."
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Choose the most appropriate tool based on the user's question.",
        ],
        extra_instructions=extra_instructions,
        storage=PgAssistantStorage(table_name="agent_runs", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="users",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
            ),
            num_documents=3,
        ),
        tools=tools,
        team=team,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        add_chat_history_to_messages=True,
        num_history_messages=4,
        markdown=True,
        add_datetime_to_instructions=True,
        introduction=dedent(
            """\
            Hi, I'm Afro Prime v7, your powerful AI Assistant. Send me on my mission boss :statue_of_liberty:\
            """
        ),
        debug_mode=debug_mode,
    )
    return agent




agent = get_agent()
question1 = "请问3+3等于多少？"
# question1 = "请查询表格中关于煤（coal）的信息，并分析表格内容"
# question2 ="请将’我爱祖国11‘这几个字保存到文件夹中"
agent.print_response(question1)

