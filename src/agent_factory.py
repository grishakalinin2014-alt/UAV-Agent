import config
from smolagents import CodeAgent
from src.tools import (
    resolve_objects_from_query,
    get_folder_by_query,
    visualize_keypoints_from_image,
    uav_simulation,
)
from src.tools_Gemini import (
    describe_satellite_image_Gemini,
    pixelpoint_objects_Gemini,
    detect_and_display_Gemini
)
from src.data_loader import (
    load_local_images,
    fetch_images_from_github
)

SYSTEM_PROMPT = """
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:
{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
{%- endfor %}

{%- if managed_agents and managed_agents.values() | list %}
You can also give tasks to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task', a long string explaining your task.
Given that this team member is a real human, you should be very verbose in your task.
Here is a list of the team members that you can call:
{%- for agent in managed_agents.values() %}
- {{ agent.name }}: {{ agent.description }}
{%- endfor %}
{%- endif %}

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict, use them directly.
4. Take care to not chain too many sequential tool calls in the same code block.
5. Call a tool only when needed, and never re-do a tool call with the exact same parameters.
6. Don't name any new variable with the same name as a tool.
7. Never create any notional variables in our code.
8. You can use imports in your code, but only from authorized list.
9. The state persists between code executions.
10. Don't give up! You're in charge of solving the task.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

def create_uav_agent(model):
    """Агент БПЛА: Крутит симуляцию и анализирует кадры на угрозы"""
    agent = CodeAgent(
        tools=[uav_simulation, detect_and_display_Gemini, fetch_images_from_github],
        model=model,
        max_steps=6,
        verbosity_level=1,
        additional_authorized_imports=["json", "PIL", "matplotlib.pyplot", "typing", "numpy", "cv2"],
        name="uav_agent",
        description="Subordinated agent. Use it to deploy the UAV flight simulation over specific keypoints and execute automated threat/fire detection on the frame stream."
    )
    agent.prompt_templates['system_prompt'] = SYSTEM_PROMPT
    return agent

def create_airspace_manager(model, uav_agent):
    """Главный менеджер: Читает запрос, ищет точки на карте, передает задачу на БПЛА"""
    agent = CodeAgent(
        tools=[
            resolve_objects_from_query,
            get_folder_by_query,
            pixelpoint_objects_Gemini,
            visualize_keypoints_from_image,
            fetch_images_from_github
        ],
        model=model,
        max_steps=12,
        verbosity_level=3,
        planning_interval=3,
        additional_authorized_imports=["json", "PIL", "matplotlib.pyplot", "io", "imageio", "os"],
        managed_agents=[uav_agent]
    )
    agent.prompt_templates['system_prompt'] = SYSTEM_PROMPT
    return agent