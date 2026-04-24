import os
import subprocess

# Cell 1
def setup_colab_environment():
    subprocess.run(["pip", "install", "gdown", "--upgrade"])
    if os.path.isdir("/content/drive/MyDrive/colab-sg2-ada-pytorch"):
        os.chdir("/content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch")
    elif os.path.isdir("/content/drive/"):
        os.chdir("/content/drive/MyDrive/")
        if not os.path.isdir("colab-sg2-ada-pytorch"):
            subprocess.run(["mkdir", "colab-sg2-ada-pytorch"])
        os.chdir("colab-sg2-ada-pytorch")
        if not os.path.isdir("stylegan2-ada-pytorch"):
            subprocess.run(["git", "clone", "https://github.com/dvschultz/stylegan2-ada-pytorch"])
        os.chdir("stylegan2-ada-pytorch")
        for dir_name in ["downloads", "datasets", "pretrained"]:
            if not os.path.isdir(dir_name):
                subprocess.run(["mkdir", dir_name])
        subprocess.run(["gdown", "--id", "1-5xZkD8ajXw1DdopTkH_rAoCsD72LhKU", "-O", "/content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/pretrained/wikiart.pkl"])
    else:
        subprocess.run(["git", "clone", "https://github.com/dvschultz/stylegan2-ada-pytorch"])
        os.chdir("stylegan2-ada-pytorch")
        for dir_name in ["downloads", "datasets", "pretrained"]:
            if not os.path.isdir(dir_name):
                subprocess.run(["mkdir", dir_name])
        os.chdir("pretrained")
        subprocess.run(["gdown", "--id", "1-5xZkD8ajXw1DdopTkH_rAoCsD72LhKU"])
        os.chdir("../")

setup_colab_environment()

# Cell 2
subprocess.run(["pip", "uninstall", "jax", "jaxlib", "-y"])
subprocess.run(["pip", "install", "jax[cuda11_cudnn805]==0.3.10", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"])
#subprocess.run(["pip", "uninstall", "torch", "torchvision", "-y"])
#subprocess.run(["pip", "install", "1.9.0+cu102", "torchvision==0.10.0+cu102", "-f", "https://download.pytorch.org/whl/torch_stable.html"])
#subprocess.run(["pip", "install", "timm==0.4.12", "ftfy==6.1.1", "ninja==1.10.2", "opensimplex"])

# Cell 3
dataset_path = '/content/drive/MyDrive/dor1.zip'
resume_from = '/content/drive/MyDrive/sonic/network-snapshot-000016.pkl'
aug_strength = 0.257
train_count = 0
mirror_x = True

# Optional
gamma_value = 50.0
augs = 'bg'
config = '11gb-gpu'
snapshot_count = 4

# Cell 4
command = [
    "python", "train.py",
    "--gpus=1",
    "--cfg=" + config,
    "--metrics=None",
    "--outdir=./results",
    "--data=" + dataset_path,
    "--snap=" + str(snapshot_count),
    "--resume=" + resume_from,
    "--augpipe=" + augs,
    "--initstrength=" + str(aug_strength),
    "--gamma=" + str(gamma_value),
    "--mirror=" + str(mirror_x),
    "--mirrory=False",
    "--nkimg=" + str(train_count)
]

subprocess.run(command)
