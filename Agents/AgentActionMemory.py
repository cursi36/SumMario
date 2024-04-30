from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os

def parseActionToolOutput(agent_output):
    agent_info = agent_output[0]
    tool_name = agent_info.tool
    tool_input = agent_info.tool_input
    agent_message = agent_output[1]

    # output_str = f""" The output from TOOL:'{tool_name}' with TOOL_INPUT:'{tool_input}' is:
    # OUTPUT:'{agent_message}'"""

    # return output_str
    return agent_message

class AgentActionMemory(ConversationBufferWindowMemory):

    def save_context(self, inputs, outputs) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)

        for agent_output in outputs['intermediate_steps']:
            output_str_ = parseActionToolOutput(agent_output)

            self.chat_memory.add_ai_message(output_str_)