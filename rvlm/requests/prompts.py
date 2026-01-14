import textwrap

def get_obj_keys_prompt(task: str):
    prompt = textwrap.dedent("""\
    List the object parts (e.g., top, bottom, handle, ...) relevant to the task:  {{TASK}}

    The answer should follow the JSON format:
    [{"key": <key>}, ...]
    """)
    prompt = prompt.replace("{{TASK}}", task)
    return prompt

def get_pointing_prompt(task: str, gripper: bool = True):
    # prompt = textwrap.dedent("""\
    # Point to the left robotic gripper and at all objects relevant to the task: {{TASK}}

    # The answer should follow the JSON format:
    # [{"point": <point>, "label": <object_name>_<label_0>}, ...]

    # The points are in [y, x] format normalized to 0-1000.""")

    if gripper:
        prompt = textwrap.dedent("""\
        Point to the robotic gripper and the object parts (e.g., top, bottom, handle, ...) relevant to the task: {{TASK}}

        The answer should follow the JSON format:
        [{"point": [<point>], "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")
    else:
        prompt = textwrap.dedent("""\
        Point to object parts (e.g., front, back,top, bottom, handle, tip, center, ...) relevant to the task: {{TASK}}

        The answer should follow the JSON format:
        [{"point": [<point>], "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")

    prompt = prompt.replace("{{TASK}}", task)

    return prompt

def get_alternative_pointing_prompt(task: str, keys: list, gripper: bool = True):
    # prompt = textwrap.dedent("""\
    # Point to the left robotic gripper and at all objects relevant to the task: {{TASK}}

    # The answer should follow the JSON format:
    # [{"point": <point>, "label": <object_name>_<label_0>}, ...]

    # The points are in [y, x] format normalized to 0-1000.""")

    if gripper:
        prompt = textwrap.dedent("""\
        Point to the robotic gripper and {{KEYS}} relevant to the task: {{TASK}}

        The answer should follow the JSON format:
        [{"point": [<point>], "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")
    else:
        prompt = textwrap.dedent("""\
        Point to {{KEYS}} relevant to the task: {{TASK}}

        The answer should follow the JSON format:
        [{"point": [<point>], "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")

    prompt = prompt.replace("{{TASK}}", task)
    prompt = prompt.replace("{{KEYS}}", str(keys))

    return prompt

def get_alternative_trajectory_prompt(task: str, points_dict: dict, n_points: int, gripper: bool = True):
    if gripper:
        prompt = textwrap.dedent("""\
        List {{N}} points for the trajectory each object in {{OBJECTS}} must follow to complete {{TASK}}.

        You are given the following initial points {{POINTS}}

        The points should be labeled by order of the trajectory, from '0' (start
        point at left hand) to <n> (final point).

        The answer should follow the JSON format:
        [{"points": [<point_0>, <point_1>, ...], "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")
    else:
        prompt = textwrap.dedent("""\
        List {{N}} points for the trajectory each object in {{OBJECTS}} must follow to complete {{TASK}}.
        Ensure the objects don't collide on their way and end up at the goal.

        You are given the following initial points {{POINTS}}

        The points should be labeled by order of the trajectory, from '0' (start
        point at left hand) to <n> (final point).

        The answer should follow the JSON format:
        [{"points": [<point_0>, <point_1>, ...], "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")
        
    prompt = prompt.replace("{{TASK}}", task)

    OBJECTS = [p["label"] for p in points_dict]
    prompt = prompt.replace("{{OBJECTS}}", str(OBJECTS))
    prompt = prompt.replace("{{POINTS}}", str(points_dict))

    prompt = prompt.replace("{{N}}", str(n_points))

    return prompt

def get_trajectory_prompt(task: str, points_dict: dict, n_points: int, gripper: bool = True):
    if gripper:
        prompt = textwrap.dedent("""\
        Place a point on the left robot gripper, then a list of {{N}} for the trajectory each object in {{OBJECTS}} must follow to complete {{TASK}}.

        You are given the following initial points {{POINTS}}

        The points should be labeled by order of the trajectory, from '0' (start
        point at left hand) to <n> (final point).

        The answer should follow the JSON format:
        [{"point": <point>, "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")
    else:
        prompt = textwrap.dedent("""\
        List {{N}} points for the trajectory each object in {{OBJECTS}} must follow to complete {{TASK}}.
        Ensure the objects don't collide on their way and end up at the goal.

        You are given the following initial points {{POINTS}}

        The points should be labeled by order of the trajectory, from '0' (start
        point at left hand) to <n> (final point).

        The answer should follow the JSON format:
        [{"point": <point>, "label": <object_name>_<label_0>}, ...]

        The points are in [y, x] format normalized to 0-1000.""")
        
    prompt = prompt.replace("{{TASK}}", task)

    OBJECTS = [p["label"] for p in points_dict]
    prompt = prompt.replace("{{OBJECTS}}", str(OBJECTS))
    prompt = prompt.replace("{{POINTS}}", str(points_dict))

    prompt = prompt.replace("{{N}}", str(n_points))

    return prompt

def get_trajectory_in_context_prompt(task: str, points_dict: dict, n_points: int, example: str):
    prompt = textwrap.dedent("""\
    Place a point on the robot gripper, then a list of {{N}} for the trajectory each object in {{OBJECTS}} must follow to complete {{TASK}}.

    You are given the following initial points {{POINTS}}

    The points should be labeled by order of the trajectory, from '0' (start
    point at left hand) to <n> (final point).

    The answer should follow the JSON format:
    [{"point": <point>, "label": <object_name>_<label_0>}, ...]

    Here's an example:
    {{EXAMPLE}}

    The points are in [y, x] format normalized to 0-1000.""")

    prompt = prompt.replace("{{TASK}}", task)

    OBJECTS = [p["label"] for p in points_dict]
    prompt = prompt.replace("{{OBJECTS}}", str(OBJECTS))
    prompt = prompt.replace("{{POINTS}}", str(points_dict))

    prompt = prompt.replace("{{EXAMPLE}}", example)

    prompt = prompt.replace("{{N}}", str(n_points))

    return prompt