#
from __future__ import annotations

from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers.json import parse_json_markdown

import os
from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from AgentActionMemory import parseActionToolOutput,AgentActionMemory
import sys
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

sys.path.append("../")
from tools_LangChain import *
from LangChainPrompts import *

from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
import os

class OutputParser(ConvoOutputParser):

    instruction_format:str = None

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.instruction_format

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into an AgentAction or AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        try:
            # Attempt to parse the text into a structured format (assumed to be JSON
            # stored as markdown)
            response = parse_json_markdown(text)

            # If the response contains an 'action' and 'action_input'
            if "action" in response and "action_input" in response:
                action, action_input = response["action"], response["action_input"]

                # If the action indicates a final answer, return an AgentFinish
                if action == "Final Answer":
                    return AgentFinish({"output": action_input}, text)
                else:
                    # Otherwise, return an AgentAction with the specified action and
                    # input
                    return AgentAction(action, str(action_input), text)
            else:
                # If the necessary keys aren't present in the response, raise an
                # exception
                raise OutputParserException(
                    f"Missing 'action' or 'action_input' in LLM output: {text}"
                )
        except Exception as e:
            # If any other exception is raised during parsing, also raise an
            # OutputParserException
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "conversational_chat"

class BaseToolsAgent():
    def __init__(self,tools,prompts_kwargs=None,llm_kwargs=None,memory_kwargs=None):

        self.memory = None
        if memory_kwargs is not None:
            memory_class = memory_kwargs.get("memory_class",AgentActionMemory)
            del memory_kwargs["memory_class"]

            self.memory = memory_class(**memory_kwargs)

        self.llm = ChatOpenAI(**llm_kwargs)

        self.tools = tools

        self.output_parser = OutputParser()
        self.output_parser.instruction_format = prompts_kwargs.get("format_instructions",FORMAT_INSTRUCTIONS)

        self.prompts_kwargs = prompts_kwargs

        self.initagent()

    def initagent(self):
        self.agent = initialize_agent(
            agent="chat-conversational-react-description",
            tools=self.tools,
            llm=self.llm,
            return_intermediate_steps=True,
            memory=self.memory,
            max_iterations=2 * len(self.tools),
            verbose=True,
            early_stopping_method='generate',
            agent_kwargs={
                'memory': self.memory,
                'output_parser': self.output_parser,
                'system_message': self.prompts_kwargs.get('prefix',PREFIX),
                'format_instructions': self.prompts_kwargs.get("format_instructions",FORMAT_INSTRUCTIONS),
                'human_message': self.prompts_kwargs.get('suffix',SUFFIX),
                'template_tool_response': self.prompts_kwargs.get('template_tool_response',TEMPLATE_TOOL_RESPONSE)
            }
        )

    def parseOutput(self,res):
        output_str = ""
        for agent_output in res['intermediate_steps']:
            output_str_ = parseActionToolOutput(agent_output)
            output_str = output_str+"\n"+output_str_
        return output_str

    def run(self,message,to_text=True):

        res = self.agent(message)
        #Add parsing

        if to_text:
            res = self.parseOutput(res)

        return res


class ToolUserAgent(BaseTool):
    name = "ToolUserAgent"
    description = """The agent responsible for using different tools.
    Input type: string #the user message
    Output type: string #the tool response
    """
    agent: BaseToolsAgent = None

    def _run(self,message):
        # to search

        res = self.agent.run(message, to_text=True)

        text = f"""The agent returned the following results:
        {res}"""
        return text

    def __call__(self, *args, **kwargs):
        return self.run(*kwargs.values())


def create_BaseToolsAgent():
    tools = [WebPageTextExtractor(), FileReader(), WebVideoTextExtractor(), GoogleWebSearcher()]

    memory_kwargs = dict(
        memory_class=AgentActionMemory,
        memory_key='chat_history',
        k=10,
        return_messages=True,
        output_key="output")

    llm_kwargs = dict(openai_api_key=os.environ['OPENAI_API_KEY'],
                      temperature=0,
                      model_name="gpt-3.5-turbo", )

    prompts_kwargs = {'prefix': PREFIX,
                      'format_instructions': FORMAT_INSTRUCTIONS,
                      'suffix': SUFFIX,
                      'template_tool_response': TEMPLATE_TOOL_RESPONSE
                      }
    return BaseToolsAgent(tools=tools, prompts_kwargs=prompts_kwargs, memory_kwargs=memory_kwargs,
                           llm_kwargs=llm_kwargs)

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from tools_LangChain import *
    from LangChainPrompts import *

    Agent = create_BaseToolsAgent()
    # Agent.run("search on google news on bitcoin today ")


    #-----LangChain init:
    from langchain.chains.conversation.memory import ConversationBufferWindowMemory

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=10,
        return_messages=True,
        output_key="output"
    )

    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        temperature=0,
        model_name="gpt-3.5-turbo",
    )

    tool = ToolUserAgent()
    tool.agent = Agent

    tools = [tool]

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        # agent='zero-shot-react-description',
        # agent = "structured-chat-zero-shot-react-description".lower(),
        tools=tools,
        llm=llm,
        return_intermediate_steps=True,
        memory=conversational_memory,
        max_iterations=2 * len(tools),
        verbose=True,
        early_stopping_method='generate',
        # output_keys=["output","intermediate_steps"],
        agent_kwargs={
            'memory': conversational_memory,
            # 'output_parser': outputParser,
            'system_message': PREFIX,
            'format_instructions': FORMAT_INSTRUCTIONS,
            'human_message': SUFFIX,
            'template_tool_response': TEMPLATE_TOOL_RESPONSE
        }
    )

    response = agent({'input':f"""Summarize the content in 'https://www.bbc.com/news/world-middle-east-68464207' 
        and in 'https://www.youtube.com/watch?v=vu2qOdS6z6c'"""})
    print(response)