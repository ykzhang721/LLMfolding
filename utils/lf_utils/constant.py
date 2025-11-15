__all__ = ['DATASET_SPLIT', 'DATASET_RAW_ROOT']

import pdb


DATASET_SPLIT = {
    # for evaluation
    'dev':              0,
    'cameo2022':        1,
    'casp15':           2,
    'casp16':           3,
    # for training
    'rcsb':             4,
    'afdb_swissprot':   5,
    'afdb_plddt90':     6,
}


DATASET_RAW_ROOT = {
    'dev':              ('/GenSIvePFS/users/lutianyu/lf/data/raw/rcsb',                 '.cif'),
    'cameo2022':        ('/GenSIvePFS/users/lutianyu/lf/data/raw/rcsb',                 '.cif'),
    'casp15':           ('/GenSIvePFS/users/lutianyu/lf/data/raw/casp/casp15',          '.pdb'),
    'casp16':           ('/GenSIvePFS/users/lutianyu/lf/data/raw/casp/casp16',          '.pdb'),
    'rcsb':             ('/GenSIvePFS/users/lutianyu/lf/data/raw/rcsb',                 '.cif'),
    'afdb_swissprot':   ('/GenSIvePFS/users/lutianyu/lf/data/raw/afdb_swissprot',       '.cif.gz'),
    'afdb_plddt90':     ('/GenSIvePFS/users/lutianyu/lf/data/raw/afdb_plddt90',         '.cif.gz'),
}

GT_STRUCT_ROOT = "/GenSIvePFS/users/data/Protein_Ref_Pred_Pair/Pair_Data_V1.1/Ref"