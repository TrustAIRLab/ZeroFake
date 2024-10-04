### Environment

You first need to build the environment by:
```
apt update && apt install -y libsm6 libxext6
conda env create -f env.yaml
conda activate zerofake
```

You also need to download the spacy model by:

```
python -m spacy download en_core_web_sm
```

### Reconstruction

You can reconstruct the given image by:

```
python uni-ddim-inversion.py --target image-path --output output-path
```

The you can compute the similarity between the origianl images and the reconstructed images by:

```
python sim.py --orginal image-path1 --reconstruct image-path2 
```

