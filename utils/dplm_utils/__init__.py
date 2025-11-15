import sys
sys.path.append("/GenSIvePFS/users/lutianyu/lf/utils/dplm_utils/dplm/src")
sys.path.append("/GenSIvePFS/users/lutianyu/lf/utils/dplm_utils/esm")
sys.path.append("/GenSIvePFS/users/lutianyu/lf/utils/dplm_utils/openfold")
from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
from byprot.models.structok.structok_lfq import VQModel
