import re
import json

import asyncio
import textwrap
import numpy as np

from rvlm.functions.function_helpers import call_gemini

def _parse_json(input: str | list) -> str:

    if type(input) == list:
        return json.dumps(input)

    elif type(input) == str:

        # parse json
        try:
            return json.loads(input)
        # extract json str, then parse
        except:
            code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
            match = code_block_pattern.search(input)
            if match:
                return match.group(1).strip()

            array_pattern = re.compile(r"\[\s*\{[\s\S]*?\}\s*\]")
            match = array_pattern.search(input)
            if match:
                return match.group(0).strip()

    return None

async def _fix_json(input: str) -> str:
    prompt = textwrap.dedent("""\
        1. Fix the JSON format of the following input:
        {input}
        2. Return the fixed JSON format.
    """)
    prompt = prompt.replace("{input}", input)
    res = await call_gemini(prompt, thinking_level="LOW")
    return res.text

# 1. image -> labels + points
async def get_labels_and_points(image: np.ndarray) -> str:
    prompt = textwrap.dedent("""\
        1. Describe the scene in great detail. Keep it concise (250 words).
        
        2. List the objects AND (multiple) descriptors (e.g., top, bottom, handle, center, left, right, blue, green, transparent, ...).

        3. Ground your description in the image, and conclude with the following JSON format:
        [{"point": [<point>], "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.
    """)

    res = await call_gemini(prompt, img_input=image, thinking_level="LOW")
    return res.text

# 2. labels + image -> scene description + points]
async def get_descriptions_and_points(video: np.ndarray, labels: list[str]) -> list[str]:
    prompt = textwrap.dedent("""\
        1. Describe the scene in great detail. Keep it concise (250 words).
        
        2. Given this list of object descriptors: {labels}, ground your description in the image, and conclude with the following JSON format:
        [{"point": [<point>], "label": <object_descriptor_0>}, ...]

        The points are in [y, x] format normalized to 0-1000. If the object is not visible, set point to [-1, -1].
        Only use the predefined object descriptors in the list!
    """)
    prompt = prompt.replace("{labels}", str(labels))

    res = await asyncio.gather(*[call_gemini(prompt, img_input=img, thinking_level="LOW") for img in video])
    
    return [r.text for r in res]

# 2. labels + task -> subtasks + target points + constraints
async def get_subtasks(image: np.ndarray, task: str, labels: list[str]) -> str:
    prompt = textwrap.dedent("""\

        Task: "{task}"
        List of object descriptors: {labels}

        1. Provide granular (e.g., pick-and-place are two separate sub-tasks) list of relevant sub-tasks the robot must complete to solve the following task.
        2. Given the list of object descriptors, define a minimize distance constraint between two object descriptors that is required to complete the subtask.
        3. Ground the subtasks in the image by providing the target points of the two objects and descriptors.
        4. Don't include free-space motions like "moving" or "reaching".

        Your answer should conclude with the following JSON format:
        ```json
        [{"sub_task_1": str, "constraint": [<object_descriptor_0>, <object_descriptor_1>], "points": [<point_0>, <point_1>]}, ...]
        ```

        If the task cannot completed within the scene due to missing objects, return:
        ```json
        []
        ```

        Only use the predefined object descriptors in the list!
    """)
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{labels}", str(labels))

    res = await call_gemini(prompt, img_input=image, thinking_level="MEDIUM")
    return res.text

# 3. subtasks + scene description(s) -> verification
async def get_verification(subtasks: list[str], scene_descriptions: list[str]) -> str:
    prompt = textwrap.dedent("""\

        Given the list of sub-tasks:
        "{subtasks}"

        and temporal scene description of what happened:
        "{scene_descriptions}"

        provide success/failure and exact timestamps when the sub-task was successful (or -1 for failures) for each of the sub-tasks in the following JSON format:
        ```json
        [{"sub_task_1": str, "success": bool, "time": int}, {"sub_task_2": str, "success": bool, "time": int}, ...]
        ```

        Some descriptions might be wrong, average or smooth out incorrect descriptions over multiple timesteps!
    """)
    prompt = prompt.replace("{subtasks}", str(subtasks))
    prompt = prompt.replace("{scene_descriptions}", str(scene_descriptions))

    res = await call_gemini(prompt, thinking_level="MEDIUM")
    return res.text

async def compute_rewards(language_instruction: str, video: np.ndarray) -> tuple[list[float], list[float]]:
    
    # 1. image -> labels + points
    result = await get_labels_and_points(video[0])
    labels_and_points = json.loads(_parse_json(result))
    # text -> list[str]
    labels = [lp["label"] for lp in labels_and_points]
    print("1. LABELS\n", labels)

    # result = await asyncio.gather(
    #     # 2. labels + task -> subtasks + target points + constraints
    #     get_subtasks(video[0], language_instruction, labels),
    #     # 2. labels + image -> scene description + points
    #     get_descriptions_and_points(video, labels),
    # )

    result = []
    result.append(await get_subtasks(video[0], language_instruction, labels))
    print("2. SUBTASKS\n", result[0])

    # text -> list[dict]
    try:
        subtasks = json.loads(_parse_json(result[0]))
    except Exception as e:
        print(e, "--> fixing json w/ Gemini")
        # gemini sometimes drops closing '}'
        def _repair_truncated_json(s: str) -> str:
            # If the last object is missing its closing brace
            s = s.strip()
            if s.endswith("]]") or s.endswith("]\n]"):
                s = s[:-1] + "}]"
            return s
        # fixed_json = await _fix_json(result[1])
        fixed_json = _repair_truncated_json(result[0])
        print("FIXED JSON\n", fixed_json)
        subtasks = json.loads(_parse_json(fixed_json))
    # list[dict] -> dict{str: str}
    subtasks_list = {k:v for r in subtasks for k,v in r.items() if "sub_task" in k}

    # early stopping
    progress = np.zeros(len(video))
    subtask_progress = np.zeros(len(video))
    if not subtasks:
        return progress, subtask_progress

    # NOTE: for matrix computation, we save calls by prematurely exiting once we know there are no subtasks
    result.append(await get_descriptions_and_points(video, labels))
    print("2. DESCRIPTIONS\n", result[1])

    # list[str] -> dict{int: str}
    scene_descriptions = {t: v for t, v in enumerate(result[1])}
    

    if subtasks:
        # 3. subtasks + scene description(s) -> verification
        result = await get_verification(subtasks_list, scene_descriptions)
        # text -> list[dict]
        verification = json.loads(_parse_json(result))
    else:
        verification = []
    print("3. VERIFICATION\n", verification)

    # early stopping
    progress = np.zeros(len(video))
    subtask_progress = np.zeros(len(video))
    if not subtasks:
        return progress, subtask_progress

    # misc
    tracked_points = {}
    for k in labels:
        tracked_points[k] = []
        for sd in scene_descriptions.values():
            
            # p = [s["point"] for s in json.loads(_parse_json(sd)) if s["label"] == k]
            # tracked_points[k].append(p[0])
            p = [s["point"] for s in json.loads(_parse_json(sd)) if s["label"] == k]
            tracked_points[k].append(p[0] if p else [-1, -1])

        tracked_points[k] = np.array(tracked_points[k])

    # 4. compute tracking progress

    # TODO: sort verfication by order of sub-tasks completion
    # verification = sorted(verification, key=lambda x: x["time"] if x["time"] != -1 else float("inf"))
    
    prev_terminal = 0
    for i, (sub, ver) in enumerate(zip(subtasks, verification)):

        terminal = ver["time"]

        if terminal == -1:
            continue

        # TODO: instead of skipping, set terminal to end of video to apply tracking for first failed sub-task
        # if terminal == -1:
        #     terminal = len(video)

        sub_constraints = sub["constraint"]
        sub_target_points = sub["points"]

        source_key = sub_constraints[0]
        target_point = np.array(sub_target_points[1])

        # track distance of constraint to target_point
        distances = np.linalg.norm(tracked_points[source_key] / 1000 - target_point / 1000, axis=1)

        distances = (1 - distances)

        progress[prev_terminal:terminal] = distances[prev_terminal:terminal]
        prev_terminal = terminal

    progress = np.cumsum(progress) / len(video)

    # 4. compute subtask progress
    for v in verification:
        if v["time"] > 1:
            subtask_progress[v["time"]] = 1
    subtask_progress = np.cumsum(subtask_progress) / len(subtasks)

    return progress, subtask_progress