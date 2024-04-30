PREFIX = """The user will ask you to complete a task. You MUST choose what function tools to use.
Your goal is to only properly call the tools.
You MUST NOT add any additional text to the one output from the tools.
"""

FORMAT_INSTRUCTIONS = """
RESPONSE FORMAT INSTRUCTIONS

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names} or Final Answer

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": "$TOOL_NAME",
  "action_input": "$INPUT"
}}}}
```

ALWAYS use the following steps:

Thought: #consider previous steps and choose the next action to take (use tool or Final Answer)
Action:
```
$JSON_BLOB
```
Observation: #the output from the tool

You MUST only return the output from the tool. No additional information.

You can repeat the steps if multiple tools are needed to complete the user's command.
You MUST return Final Answer if no more tools are needed. For Final Answer the response MUST be:
```
{{{{
  "action": "Final Answer",
  "action_input": "Done"
}}}}
```

"""

SUFFIX = """TOOLS
------
The tools to use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}"""

TEMPLATE_TOOL_RESPONSE = """ The tool has produced the following output: 
{observation}.

Tools MUST NOT be used with temporary txt files.
Choose if another tool is needed or return Final Answer.

"""