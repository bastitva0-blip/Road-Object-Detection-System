from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="road-object-detection",
    version="0.1.0",
    author="Robotics Club",
    description="A comprehensive OpenCV + AI-powered real-time road awareness system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "mediapipe>=0.10.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pyyaml>=6.0.0",
    ],
)
