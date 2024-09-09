from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fr:
    requirements = fr.read().splitlines()

def is_arm_cpu():
    """
    Check if the CPU architecture is ARM.

    This function determines whether the current system is running on
    an ARM CPU by inspecting the machine architecture string provided
    by the platform module.

    Returns:
        bool: True if the CPU is ARM, False otherwise.
    """
    import platform
    architecture = platform.machine()
    return architecture.startswith('arm') or architecture.startswith('aarch')

if not is_arm_cpu():
    requirements.append("tensorflow>=2.13.0")

setup(
    name='EfficientWord-Net',
    version='1.1.1',
    description='Few Shot Learning based Hotword Detection Engine',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Ant-Brain/EfficientWord',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)