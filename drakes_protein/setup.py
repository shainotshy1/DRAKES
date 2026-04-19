import setuptools as tools

tools.setup(
    name="multiflow",
    packages=[
        'openfold',
        'multiflow',
        'ProteinMPNN',
        'protein_oracle'
    ],
    package_dir={
        'openfold': './openfold',
        'multiflow': './multiflow',
        'ProteinMPNN': './ProteinMPNN',
        'protein_oracle': './protein_oracle',
    },
)
