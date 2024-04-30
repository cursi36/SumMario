import sys
sys.path.append("./")
from BaseToolsAgent import BaseToolsAgent,create_BaseToolsAgent,OutputParser
from ConversationAgent import ConversationAgent,create_ConversationAgent

from AgentActionMemory import parseActionToolOutput,AgentActionMemory
from langchain.tools import BaseTool

sys.path.append("../")
from tools_LangChain import *
from LangChainPrompts import *

import os


def parsefilename(filename):
    if os.getcwd() in filename:
        return filename
    else:
        return os.path.join(os.getcwd(), "tmp_files/" + filename)



class DataRetriever(BaseTool):
    name = "DataRetriever"
    description = """The tool responsible for extracting text from files, webpages, videos, and searching online.
    It MUST NOT be used with temporary txt files.
    It saves the data results in a temporary txt file.
    Input type: string #the user message
    Output type: string #the name of the saved file.
    """
    agent: BaseToolsAgent = None

    def _run(self,message):
        # to search

        res = self.agent.run(message, to_text=True)

        return res

    def __call__(self, *args, **kwargs):
        return self.run(*kwargs.values())

class ConversationalAgent(BaseTool):
    name = "ConversationalAgent"
    description = """Large language model based agent for completing a task from retrieved data and the given user message.
    It can directly process temporary txt files.
    Input type: List of strings # "[the user message,the name of the temporary file to read or "None"]".
    Output type: string #the tool response
    """
    agent: ConversationAgent = None

    def read_txt(self,filename):
        try:
            with open(filename, 'r',encoding="utf-8") as file:
                text = file.read()
        except:
            with open(filename, 'r') as file:
                text = file.read()

        return text


    def parse_message_and_filename(self,message):
        message_ = (message.replace("[", "")).replace("]", "")
        split_tries = ['",', "',"]
        for split_ in split_tries:
            message_list = message_.split(split_)
            if len(message_list) > 1:
                break

        user_message = message_list[0]
        filename = message_list[1].replace("'", "").strip()
        try:
            if filename is not None and 'none' not in filename.lower():
                filename = parsefilename(filename)
                text = self.read_txt(filename)
                user_message = user_message + f"\n You MUST use the following text: ```{text}```"
        except:
            pass

        return user_message

    def _run(self,message):
        message_ = self.parse_message_and_filename(message)

        res = self.agent.run(message_, to_text=True)

        return res

    def __call__(self, *args, **kwargs):
        return self.run(*kwargs.values())

class SuperAgent(BaseToolsAgent):
    def __init__(self, tools, prompts_kwargs=None, llm_kwargs=None, memory_kwargs=None):
        super().__init__(tools, prompts_kwargs, llm_kwargs, memory_kwargs)

    def parseOutput(self,res):
        output_str = ""
        for agent_output in res['intermediate_steps']:
            output_str_ = parseActionToolOutput(agent_output)
            output_str = output_str+"\n"+output_str_
        return output_str

    def run(self,message,to_text=True):

        message = message+"\n Should you use DataRetriever or ConversationalAgent?"
        res = self.agent(message)
        #Add parsing

        if to_text:
            res = self.parseOutput(res)

        return res

def create_SuperAgent():
    ToolAgent = DataRetriever()
    ToolAgent.agent = create_BaseToolsAgent()

    ConvAgent = ConversationalAgent()
    ConvAgent.agent = create_ConversationAgent()

    superagent_tools = [ToolAgent, ConvAgent]

    memory_kwargs = dict(
        memory_class=AgentActionMemory,
        memory_key='chat_history',
        k=10,
        return_messages=True,
        output_key="output")

    llm_kwargs = dict(openai_api_key=os.environ['OPENAI_API_KEY'],
                      temperature=0,
                      model_name="gpt-3.5-turbo", )

    SUPERAGENT_PREFIX = """You are an agent manager that has access to two tools with different specialities.  
        DataRetriever tool is able to extract data from external resources and do online searches,
         and ConversationalAgent processes the user message and the retrieved data.

        The user will ask you to complete a task. You MUST choose what tool to use.
        Your goal is to only properly call the tools. 
        You MUST always reason if external resources are needed first or if you can directly use the conversational tool.
        Reminder that DataRetriever MUST NOT be used with temporary txt files.
        You MUST NOT add any additional text to the one output from the tools.
        """

    prompts_kwargs = {'prefix': SUPERAGENT_PREFIX,
                      'format_instructions': FORMAT_INSTRUCTIONS,
                      'suffix': SUFFIX,
                      'template_tool_response': TEMPLATE_TOOL_RESPONSE
                      }
    return SuperAgent(tools=superagent_tools, prompts_kwargs=prompts_kwargs, memory_kwargs=memory_kwargs,
                            llm_kwargs=llm_kwargs)

if __name__ == "__main__":
    sys.path.append("./")
    superAgent = create_SuperAgent()

#     message = f"""Summarize in 10 lines the content in 'https://www.youtube.com/watch?v=vu2qOdS6z6c'
# Should you use DataRetriever or ConversationalAgent?"""

    message = f"""Hello how can you help me?"""

    message = """"
    ["What is described at the link 'https://www.geeksforgeeks.org/saving-text-json-and-csv-to-a-file-in-python/' ?", 'webpage_content_4b0e09f4-2ef6-4332-9c26-735716055495.txt']
    """

    ConvAgent = ConversationalAgent()
    ConvAgent.agent = create_ConversationAgent()
    ConvAgent.run(message)

    reposne = superAgent.run(message,to_text=True)
    print()