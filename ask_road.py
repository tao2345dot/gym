import numpy as np
from openai import OpenAI


def ask_road(init_xyz, num_waypoints=3000, clockwise=False):
    """
    使用 LLM 生成 Python 代码，计算每个无人机轨迹：
    - 所有轨迹围绕 (0,0)
    - 半径为所有 UAV 初始位置在 XY 投影拟合的圆
    - z 不变
    - 顺/逆时针旋转，轨迹点数量由 num_waypoints 控制
    - 均匀分布，不含初始点
    """
    direction_text = "clockwise" if clockwise else "counterclockwise"
    print(f"[INFO]: Requesting UAV trajectory with {num_waypoints} waypoints, direction: {direction_text}")

    client = OpenAI(
        api_key="",  # 替换为你自己的 key
        base_url="https://api.chatanywhere.tech"
    )

    if isinstance(init_xyz, np.ndarray):
        init_xyz = init_xyz.tolist()

    prompt = f"""
You are a Python expert in drone simulation.

Write Python code that defines a variable `trajectories`, which is a NumPy array of shape (N, {num_waypoints}, 3), where N is the number of drones and each element is a list of 3D positions (X, Y, Z) representing the trajectory of one drone.

Input:
- A list of initial 3D positions of drones: {init_xyz}

Requirements:
1. Fit a circle in the XY plane using all initial drone positions (ignore Z). Use numpy's least squares to solve for the circle center (cx, cy) and radius R.
2. For each drone:
   - Keep the Z-value constant (same as initial Z)
   - Compute the initial angle from the center using: angle = np.arctan2(y - cy, x - cx)
   - Generate {num_waypoints} points along the circle, evenly spaced
   - The first point must follow the initial angle (do NOT include the initial point)
   - The direction must be {'clockwise' if clockwise else 'counterclockwise'}
   - Use numpy arrays for all calculations
3. Finally, stack all drone trajectories into a single NumPy array of shape (N, {num_waypoints}, 3) and assign it to the variable `trajectories`.

⚠️ Output Format:
- Only output valid Python code
- Only define the variable `trajectories`
- No imports, no markdown, no comments, no print
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048
        )

        code = response.choices[0].message.content.strip()
        print("[DEBUG_LLM_CODE]:\n", code)

        if code.startswith("```"):
            code = code.strip("`").strip()
            if code.startswith("python"):
                code = code[len("python"):].strip()

        local_vars = {}
        exec(code, {"np": np}, local_vars)

        if "trajectories" not in local_vars:
            raise ValueError("LLM did not define 'trajectories' variable.")

        trajectories = np.array(local_vars["trajectories"])

        if trajectories.ndim != 3 or trajectories.shape[1] != num_waypoints or trajectories.shape[2] != 3:
            raise ValueError(f"Invalid trajectory shape: {trajectories.shape}")

        print(f"[INFO]: Trajectories shape: {trajectories.shape}")
        return trajectories

    except Exception as e:
        print("[ERROR]: Failed to get trajectory from LLM:", e)
        return None



