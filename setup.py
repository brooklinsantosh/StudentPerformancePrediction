from setuptools import find_packages, setup

def get_requirements(file_path: str) -> list[str]:
    """
    This function will read the requirements.txt file and return the required libraries.
    """
    HYPHEN_E_DOT = "-e ."
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        requirements = [r.replace("\n","") for r in req]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return req

setup(
    name="StudentPerformancePrediction",
    version="0.0.1",
    author="Brooklin Santosh A G S",
    author_email="brooklinsantosh@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)