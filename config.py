# -----------------
# DATASET ROOTS
# ----------------- 
# domainnet_dataroot = '/disk/work/hjwang/gcd/domainbed/data/domain_net/'
domainnet_dataroot = '/mnt/sdb/fengwei/GCD_natural/domain_dataset/domain_net/'

cub_root = '/mnt/sdb/fengwei/GCD_natural/dataset/CUB/'
cubc_root = '/mnt/sdb/fengwei/GCD_natural/domain_dataset/cub-c'

fgvc_root = '/mnt/sdb/fengwei/GCD_natural/dataset/FGVCAircraft/fgvc-aircraft-2013b'
fgvcc_root = '/mnt/sdb/fengwei/GCD_natural/domain_dataset/fgvc-c'

scars_root = '/mnt/sdb/fengwei/GCD_natural/dataset/StanfordCars/stanford_cars/cars_{}/'
scarsc_root = '/mnt/sdb/fengwei/GCD_natural/domain_dataset/scars-c'
scars_meta_path = "/mnt/sdb/fengwei/GCD_natural/dataset/StanfordCars/stanford_cars/devkit/cars_{}.mat"

# OSR Split dir
osr_split_dir = '/mnt/sdb/fengwei/GCD_natural/HiLo-main/data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/mnt/sdb/fengwei/GCD_natural/pretrain_path/dino_vitbase16_pretrain.pth' 
clip_pretrain_path = '/mnt/sdb/fengwei/GCD_natural/pretrain_path/clip/ViT-B-16.pt' 
feature_extract_dir = '/mnt/sdb/fengwei/GCD_natural/pretrain_path/extracted_features_public_impl'     # Extract features to this directory

# -----------------
# Corruption types
# -----------------
# distortions = [
#     'gaussian_noise', 'shot_noise', 'impulse_noise',
#     'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
#     'snow', 'frost', 'fog', 'elastic_transform', 'pixelate', 'jpeg_compression',
#     'speckle_noise', 'spatter'#, 'saturate'
# ]
# distortions = [
#     'gaussian_noise', 'shot_noise', 'impulse_noise',
#     'zoom_blur',
#     'snow', 'frost', 'fog',
#     'speckle_noise', 'spatter'#, 'saturate'
# ]
# severity = [1, 2, 3, 4, 5]

distortions = [
    'snow',
]
severity = [5]
# -----------------
# domain types
# -----------------
ovr_envs = ["real", "painting", "quickdraw", "sketch", "clipart", "infograph"]