from setuptools import setup

package_name = "robotic_ultrasound_force_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/digital_twin.launch.py"]),
        ("share/" + package_name + "/config", ["config/digital_twin.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="QLlaith11",
    maintainer_email="alshiekhlaith@gmail.com",
    description="MuJoCo digital-twin and force-controller interfaces.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "controller_node = robotic_ultrasound_force_control.controller_node:main",
            "mujoco_node = robotic_ultrasound_force_control.mujoco_node:main",
            "safety_node = robotic_ultrasound_force_control.safety_node:main",
        ],
    },
)
