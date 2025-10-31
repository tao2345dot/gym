import os
import json
import numpy as np


def _geometric_plan(agent_pos, obstacle_pos, radius, target_pos, clearance=0.5):
    """几何退路：与之前实现相同的简单候选点选择器"""
    agent = np.array(agent_pos, dtype=float)
    obs = np.array(obstacle_pos, dtype=float)
    target = np.array(target_pos, dtype=float)

    vec = agent - obs
    if np.linalg.norm(vec[:2]) < 1e-6:
        vec = np.array([1.0, 0.0, 0.0])

    perp = np.array([-vec[1], vec[0], 0.0])
    perp_norm = np.linalg.norm(perp[:2])
    if perp_norm < 1e-6:
        perp = np.array([0.0, 1.0, 0.0])
    else:
        perp = perp / perp_norm

    offset = radius + clearance
    cand1 = obs + perp * offset
    cand2 = obs - perp * offset

    cand1[2] = target[2]
    cand2[2] = target[2]

    d1 = np.linalg.norm(cand1 - target)
    d2 = np.linalg.norm(cand2 - target)
    chosen = cand1 if d1 < d2 else cand2
    return chosen


def plan_avoidance(agent_pos, obstacle_pos, radius, target_pos, clearance=0.5, model='gpt-3.5-turbo', env_bounds=None, other_obstacles=None, constraints=None):
    """
    使用 LLM (OpenAI) 生成避障点的包装器。

    行为：如果环境变量 OPENAI_API_KEY 可用且 openai 库可导入，则调用 ChatCompletion
    请求 LLM 返回一个 JSON 对象 {"avoid_point": [x, y, z]}。如果出现任何错误或 API Key 不存在，
    会回退到本地几何规划 `_geometric_plan`。

    注意：请通过环境变量设置 OPENAI_API_KEY（不要把 key 写进代码）。
    """
    # 优先尝试使用 OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            import openai
            openai.api_key = api_key

            # Build a richer prompt including environment bounds, other obstacles and constraints
            prompt_lines = []
            prompt_lines.append("You are a precise geometric planner. Reply with strict JSON containing an 'avoid_point' field (3 floats) or a bare list [x, y, z].")
            prompt_lines.append(f"Agent position: {list(map(float, agent_pos))}")
            prompt_lines.append(f"Primary obstacle: center={list(map(float, obstacle_pos))}, radius={float(radius):.3f}")
            prompt_lines.append(f"Original target: {list(map(float, target_pos))}")

            if env_bounds is not None:
                # env_bounds is expected to be a dict-like with x_range/y_range/z_range
                try:
                    xb = env_bounds.get('x_range')
                    yb = env_bounds.get('y_range')
                    zb = env_bounds.get('z_range')
                    prompt_lines.append(f"Environment bounds: X{xb}, Y{yb}, Z{zb}")
                except Exception:
                    prompt_lines.append(f"Environment bounds: {str(env_bounds)}")

            # other_obstacles expected as list of (pos, radius)
            # Avoid using a bare truth-value check in case a NumPy array is passed.
            if other_obstacles is not None:
                try:
                    obs_strs = []
                    for o in other_obstacles:
                        pos = list(map(float, o[0]))
                        rad = float(o[1])
                        obs_strs.append(f"center={pos},r={rad:.2f}")
                    prompt_lines.append(f"Other obstacles: {obs_strs}")
                except Exception:
                    # Fall back to a safe stringification if iteration/parsing fails
                    prompt_lines.append(f"Other obstacles: {str(other_obstacles)}")

            if constraints:
                prompt_lines.append(f"Constraints: {json.dumps(constraints)}")

            prompt_lines.append(f"Requirements: produce one safe avoidance waypoint that keeps at least (radius + {clearance:.2f}) m clearance from the primary obstacle in the XY plane, stays within environment bounds, and avoids other obstacles if possible. Keep the z coordinate near the target's z. Return ONLY valid JSON or a bare numeric list.")

            prompt = "\n".join(prompt_lines)

            messages = [
                {"role": "system", "content": "You are a helpful and precise geometric planner. Reply with strict JSON."},
                {"role": "user", "content": prompt}
            ]

            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=128
            )

            text = resp['choices'][0]['message']['content']
            # 尝试解析 JSON
            try:
                parsed = json.loads(text)
                if 'avoid_point' in parsed and isinstance(parsed['avoid_point'], list) and len(parsed['avoid_point']) >= 3:
                    return np.array(parsed['avoid_point'][:3], dtype=float)
            except Exception:
                # 如果直接是裸坐标，例如: [x, y, z]
                try:
                    parsed2 = json.loads(text.strip())
                    if isinstance(parsed2, list) and len(parsed2) >= 3:
                        return np.array(parsed2[:3], dtype=float)
                except Exception:
                    # 最后尝试从文本中提取数字
                    import re
                    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
                    if len(nums) >= 3:
                        return np.array([float(nums[0]), float(nums[1]), float(nums[2])], dtype=float)

        except Exception:
            # 任何异常都回退到几何规划
            pass

    # 回退到本地几何规划
    return _geometric_plan(agent_pos, obstacle_pos, radius, target_pos, clearance=clearance)
