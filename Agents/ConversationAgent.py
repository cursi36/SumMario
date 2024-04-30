from BaseToolsAgent import BaseToolsAgent
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os

PREFIX = """The user will ask you to complete a task. You directly reply to the human with your own reasoning.
Return Done when finished.
"""

class ConversationAgent(BaseToolsAgent):
        def __init__(self,tools,prompts_kwargs=None,llm_kwargs=None,memory_kwargs=None):
            super().__init__(tools,prompts_kwargs=prompts_kwargs,llm_kwargs=llm_kwargs,memory_kwargs=memory_kwargs)

            self.system_message = SystemMessage(
                    content=PREFIX
                )

        def parseOutput(self,res):
            return res.content

        def run(self,message,to_text=True):

            past_messages = []
            if self.memory is not None:
                past_messages = self.memory.chat_memory.messages

            messages = [self.system_message]+past_messages+[HumanMessage(content=message)]

            res = self.llm.invoke(messages, stop=["\nObservation","\n\tObservation","\nDone","\n\rDone"])

            if self.memory is not None:
                self.memory.save_context({"input":message},{"output":res.content})

            if to_text:
                res = self.parseOutput(res)

            return res


def create_ConversationAgent():
    tools = []

    memory_kwargs = dict(
        memory_class=ConversationBufferWindowMemory,
        memory_key='chat_history',
        k=10,
        return_messages=True,
        output_key="output")

    llm_kwargs = dict(openai_api_key=os.environ['OPENAI_API_KEY'],
                      temperature=0,
                      model_name="gpt-3.5-turbo", )

    return ConversationAgent(tools=tools, prompts_kwargs={}, memory_kwargs=memory_kwargs, llm_kwargs=llm_kwargs)

if __name__ == "__main__":
    convAgent = create_ConversationAgent()

    convAgent.run("Translate to italian the sentence 'Hello how are you?' ")

    print()