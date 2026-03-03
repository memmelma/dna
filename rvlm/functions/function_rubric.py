import re
import time
import json
import textwrap
import asyncio
import numpy as np
from rvlm.functions.function_helpers import call_gemini

_subtask_cache: dict = {}

def _parse_json(input) -> str:
    """
    Extract JSON from a long text similar to the frontend JS logic:
    - Prefer a ```json ... ``` fenced code block
    - Otherwise try to match an array like [... { ... }, ...]
    If not found, return an empty string.
    """

    print("RAW _parse_json input:", input)

    if type(input) == list:
        return json.dumps(input)

    elif type(input) == str:
        code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
        match = code_block_pattern.search(input)
        if match:
            return match.group(1).strip()

        array_pattern = re.compile(r"\[\s*\{[\s\S]*?\}\s*\]")
        match = array_pattern.search(input)
        if match:
            return match.group(0).strip()

    return None

# def _parse_json(text):
#     if type(text) == list:
#         return json.dumps(text)

#     print(f"[_parse_json] raw input (type={type(text).__name__}, len={len(text) if text else 0}):\n{repr(text)}")
#     if not text or not text.strip():
#         raise ValueError(f"Cannot parse empty response as JSON")
#     if type(text) == str:
#         text = text.replace("'", '"')
#     try:
#         match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
#         if match is None:
#             match = re.search(r"(.*?)```", text, re.DOTALL)
#         if match is None:
#             raise ValueError("No regex match found")
#         json_str = match.group(1).strip()
#         print(f"[_parse_json] extracted json_str: {repr(json_str)}")
#         return json.loads(json_str)
#     except Exception as e:
#         print(f"[_parse_json] regex parse failed ({e}), falling back to json.loads on full text")
#         return json.loads(text)

async def get_a(video):
    prompt = textwrap.dedent("""\
        Describe the scene in great detail. List the objects AND (multiple) descriptors (e.g., top, bottom, handle, center, left, right, blue, green, transparent, ...). Keep it concise (250 words).
    """)
    res = await asyncio.gather(*[call_gemini(prompt, img_input=img, thinking_level="LOW") for img in video])
    text_summary = {}
    for i, res in enumerate(res):
        text_summary[i+1] = res.text
        print(f"[{i+1}] DESCRIPTION\n", res.text)
    text_summary = json.dumps(text_summary)
    
    return text_summary

# async def get_b(language_instruction, video):
#     prompt = textwrap.dedent("""\
#         Point at all objects in the scene.
#     """)
#     res = await call_gemini(prompt, img_input=video[0], thinking_level="LOW")
#     objects = res.text

#     prompt = textwrap.dedent("""\

#         1. List the objects AND (multiple) descriptors (e.g., top, bottom, handle, center, left, right, blue, green, transparent, ...) that must move to complete the task. Only list minimal set and exclude unnecessary. Ground the listed objects in the image.
#         2. Provide granular list of relevant sub-tasks the robot must complete to solve the following task: "{{TASK}}"
#         3. Don't include free-space motions like "moving" or "reaching".

#         Your answer should conclude with the following JSON format:
#         ```json
#         [{"sub_task_1": str}, {"sub_task_2": str}, ...]
#         ```
#     """)

#     prompt = prompt.replace("{{TASK}}", language_instruction)

#     # call gemini
#     res = await call_gemini(prompt, img_input=video[0], thinking_level="LOW")
#     print("\nSUB-TASKS\n", res.text)

#     json_str = re.search(r"```json(.*)```", res.text, re.DOTALL).group(1)
#     json_obj = json.loads(json_str)

#     return json_obj

async def get_b(language_instruction, video):
    prompt = textwrap.dedent("""\

        Task: "{{TASK}}"

        1. List the objects AND (multiple) descriptors (e.g., top, bottom, handle, center, left, right, blue, green, transparent, ...) that must move to complete the task. Only list minimal set and exclude unnecessary. Ground the listed objects in the image.
        2. Provide granular list of relevant sub-tasks the robot must complete to solve the following task.
        3. Don't include free-space motions like "moving" or "reaching".

        Your answer should conclude with the following JSON format:
        ```json
        [{"sub_task_1": str}, {"sub_task_2": str}, ...]
        ```

        If the task cannot completed within the scene due to missing objects, return:
        ```json
        []
        ```
    """)

    prompt = prompt.replace("{{TASK}}", language_instruction)

    # call gemini
    res = await call_gemini(prompt, img_input=video[0], thinking_level="LOW")
    print("\nSUB-TASKS\n", res.text)

    if "[]" in str(res.text):
        return []

    json_str = re.search(r"```json(.*)```", res.text, re.DOTALL).group(1)
    json_obj = json.loads(json_str)

    return json_obj

async def get_c(a, b):
    prompt = textwrap.dedent("""\

        Given the list of sub-tasks:
        "{{A}}"

        and temporal scene description of what happened:
        "{{B}}"

        provide success/failure and precise timestamps (-1 for failures) for each of the sub-tasks in the following JSON format:
        ```json
        [{"sub_task_1": str, "success": bool, "time": int}, {"sub_task_2": str, "success": bool, "time": int}, ...]
        ```

        Some descriptions might be wrong, average or smooth out incorrect descriptions over multiple timesteps!
    """)

    prompt = prompt.replace("{{A}}", a)
    prompt = prompt.replace("{{B}}", b)

    # call gemini
    res = await call_gemini(prompt, thinking_level="MEDIUM")
    # print("\nCHECKLIST\n", res.text)

    return res.text
    
async def compute_language_based_subtasks(language_instruction, video):
    # a = await get_a(video)
    # b = await get_b(language_instruction)

    # WARNING: don't set subtask cache for metric computation -- multiple videos queried w/ same task description ...
    _subtask_cache = {}

    if language_instruction in _subtask_cache and len(_subtask_cache[language_instruction]) == 0:
        return []
        
    start_time = time.time()
    if language_instruction not in _subtask_cache:
        a, b = await asyncio.gather(get_a(video), get_b(language_instruction, video))        
        _subtask_cache[language_instruction] = b

    else:
        a = await get_a(video)
    b = _subtask_cache[language_instruction]
    
    if language_instruction in _subtask_cache and len(_subtask_cache[language_instruction]) == 0:
        return []

    # input is str(list of dicts)
    c = await get_c(_parse_json(b), a)
    end_time = time.time()
    print(f"Time taken for compute_language_based_subtasks: {end_time - start_time} seconds")
    print(f"[compute_language_based_subtasks] get_c raw response (type={type(c).__name__}, len={len(c) if c else 0}):\n{repr(c)}")

    # return list of dicts
    return json.loads(_parse_json(c))

def rubric_to_dense_reward(subtasks, T, delay=0):
    rewards = np.zeros(T)

    completed = [s for s in subtasks if s['success']]
    if not completed:
        return rewards

    times = [min(s['time'] + delay, T) for s in completed]

    for i, time in enumerate(times):
        rewards[time] = 1 / len(subtasks)
    
    # breakpoints = [0] + times
    # for i, (start, end) in enumerate(zip(breakpoints, breakpoints[1:] + [T])):
    #     rewards[start:end] = i + 1

    return rewards