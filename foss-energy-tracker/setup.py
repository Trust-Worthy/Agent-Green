from setuptools import setup, find_packages

setup(
    name='foss-energy-tracker',
    version='0.1.0', # Or whatever version you want
    packages=find_packages(), # This will find the 'energy_tracker' package
    install_requires=[
        'psutil',
        'numpy', # If your tracker eventually uses numpy for calculations within its own files
        # Add any other direct dependencies your tracker needs here
    ],
    author='Trust-Worthy', # Replace with your name
    author_email='jondeveloper0@gmail.com', # Replace with your email
    description='A FOSS library for tracking software energy consumption and carbon emissions.',
    long_description=open('README.md').read(), # Assuming you'll create a README.md
    long_description_content_type='text/markdown',
    url='', # Replace with your GitHub repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Or your chosen FOSS license
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Monitoring',
        'Environment :: Console',
    ],
    python_requires='>=3.8', # Minimum Python version
)