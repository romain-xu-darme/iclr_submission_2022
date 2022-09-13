import setuptools

# Read version
with open('VERSION','r') as fin:
    VERSION = fin.read()

# Read README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Setup
setuptools.setup (
    name='torch_toolbox',
    author='Anonymous',
    version=VERSION,
    license='BSD-3',
    description='Pytorch toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',

    entry_points={
        'console_scripts': [
            'torch_train_classifier = torch_toolbox.training.train_classifier:_main',
            'torch_evaluate_classifier = torch_toolbox.training.evaluate_classifier:_main',
            'torch_checkpoint_info = torch_toolbox.training.checkpoint_info:_main',
            'torch_model_summary = torch_toolbox.models.summary:_main',
        ]
    }
)
