import os
import setuptools

BASE_PATH = os.path.dirname(__file__)
with open(os.path.join(BASE_PATH, 'VERSION'), 'r') as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name="agent-exchange", # Replace with your own username
    version=VERSION,
    author="Max Holloway",
    author_email="maxwellpearmanholloway@gmail.com",
    description="Tool to simulate multi-agent interactions in competitive environments",
    long_description_content_type="text/markdown",
    url="https://github.com/maxholloway/agent-exchange",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy==1.19.2',
        'pandas==1.1.3'
    ],
    setup_requires=['wheel']
)