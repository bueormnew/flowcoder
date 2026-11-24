from setuptools import setup, find_packages

setup(
    name="flowcoder",
    version="0.1.0",
    description="Ultra-Fast, Mobile-Friendly Hybrid Generative Framework (Rectified Flow + ConvNeXt-Attention CORE)",
    author="FlowCoder Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "diffusers>=0.25.0",
        "safetensors>=0.4.0",
        "pillow",
        "numpy",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
