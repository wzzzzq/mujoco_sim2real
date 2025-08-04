from setuptools import setup, find_packages

setup(
    name="mujoco_sim2real",
    version="0.1",
    packages=find_packages(),  # 自动找到 mujoco_sim2real 及其子模块

    entry_points={
        "console_scripts": [
            # 可选：比如你想直接运行 `lerobot-train` 命令
            # "lerobot-train=lerobot.scripts.train:main"
        ]
    }
)