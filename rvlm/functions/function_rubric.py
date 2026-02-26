import re
import time
import json
import textwrap
import asyncio
import numpy as np
from rvlm.functions.function_helpers import call_gemini

_subtask_cache: dict = {}

def _parse_json(text):
    if type(text) == list:
        return json.dumps(text)
    print(f"[_parse_json] raw input (type={type(text).__name__}, len={len(text) if text else 0}):\n{repr(text)}")
    if not text or not text.strip():
        raise ValueError(f"Cannot parse empty response as JSON")
    if type(text) == str:
        text = text.replace("'", '"')
    try:
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match is None:
            match = re.search(r"(.*?)```", text, re.DOTALL)
        if match is None:
            raise ValueError("No regex match found")
        json_str = match.group(1).strip()
        print(f"[_parse_json] extracted json_str: {repr(json_str)}")
        return json.loads(json_str)
    except Exception as e:
        print(f"[_parse_json] regex parse failed ({e}), falling back to json.loads on full text")
        return json.loads(text)

async def get_a(video):
    prompt = textwrap.dedent("""\
        Describe the scene in great detail.
    """)
    res = await asyncio.gather(*[call_gemini(prompt, img_input=img, thinking_level="LOW") for img in video])
    text_summary = {}
    for i, res in enumerate(res):
        text_summary[i] = res.text
    text_summary = json.dumps(text_summary)
    # print("\nDESCRIPTION\n", text_summary)

    prompt = textwrap.dedent("""\
        Think step by step and logically.

        You're given a temporal scene description in the following form:
        {"<timestep>": <description>, ...}

        Description:
        {DESCRIPTION}

        Describe what happened. Ensure your description is temporally consistent and logical, e.g., when two entities attach (e.g., an object gets grasped), it moves with the grasping entity. Some descriptions might be wrong, average or smooth out incorrect descriptions over multiple timesteps.
    """)
    prompt = prompt.replace("{DESCRIPTION}", text_summary)

    res = await call_gemini(prompt, thinking_level="MEDIUM")
    # print("\nSUMMARY\n", res.text)

    return res.text

async def get_b(language_instruction, video):
    prompt = textwrap.dedent("""\

        Task: "{{TASK}}"
        1. List the objects AND (multiple) descriptors (e.g., top, bottom, handle, center, left, right, blue, green, transparent, ...) that must move to complete the task. Only list minimal set and exclude unnecessary. Ground the listed objects in the image.
        2. Provide list of relevant sub-tasks the robot must complete to solve the task. Ground the subtasks in the listed objects.
        3. Don't include free-space motions like "moving" or "reaching".

        Your answer should conclude with the following JSON format:
        ```json
        [{"sub_task_1": str}, {"sub_task_2": str}, ...]
        ```
    """)

    prompt = prompt.replace("{{TASK}}", language_instruction)

    # call gemini
    res = await call_gemini(prompt, img_input=video[0], thinking_level="MEDIUM")
    # print("\nSUB-TASKS\n", res.text)

    json_str = re.search(r"```json(.*)```", res.text, re.DOTALL).group(1)
    json_obj = json.loads(json_str)

    return json_obj

async def get_c(a, b):
    prompt = textwrap.dedent("""\

        Given the list of sub-tasks:
        "{{A}}"

        and temporal scene description of what happened:
        "{{B}}"

        provide success/failure and timestamps (-1 for failures) for each of the sub-tasks in the following JSON format:
        ```json
        [{"sub_task_1": str, "success": bool, "time": int}, {"sub_task_2": str, "success": bool, "time": int}, ...]
        ```
    """)

    prompt = prompt.replace("{{A}}", a)
    prompt = prompt.replace("{{B}}", b)

    # call gemini
    res = await call_gemini(prompt, thinking_level="HIGH")
    # print("\nCHECKLIST\n", res.text)

    return res.text

    # json_str = re.search(r"```json(.*)```", res.text, re.DOTALL).group(1)
    # json_obj = json.loads(json_str)

    # return json_obj
    
async def compute_language_based_subtasks(language_instruction, video):
    # a = await get_a(video)
    # b = await get_b(language_instruction)
    start_time = time.time()
    if language_instruction not in _subtask_cache:
        a, b = await asyncio.gather(get_a(video), get_b(language_instruction, video))
        _subtask_cache[language_instruction] = b
    else:
        a = await get_a(video)
    b = _subtask_cache[language_instruction]
    # a, b = await asyncio.gather(get_a(video), get_b(language_instruction, video))
    c = await get_c(_parse_json(b), a)
    end_time = time.time()
    print(f"Time taken for compute_language_based_subtasks: {end_time - start_time} seconds")
    print(f"[compute_language_based_subtasks] get_c raw response (type={type(c).__name__}, len={len(c) if c else 0}):\n{repr(c)}")
    return _parse_json(c)

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