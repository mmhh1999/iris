**TODO**

- [x] Material-geometry dataset release
- [ ] Spatially-varying lighting dataset release

**Update 2026/04**

We regret to inform users that the original dataset download link is no longer valid due to unexpected file loss on the cloud service. We sincerely apologize for the inconvenience this has caused.

We have made every effort to recover the dataset. While we were only able to restore part of the original data, the recovered subset has now been uploaded to a new server. The updated download links are available [here](Interiorverse_dataset.csv).

Please note the following details about the recovered dataset:

- The portion rendered with an 85° field of view (FOV) is largely intact.
- The portion rendered with a 120° FOV has suffered substantial data loss.

As a result of the missing data, the original train/validation/test splits are no longer valid.

If you previously downloaded the complete dataset and still have a copy, we would greatly appreciate it if you could contact [Jingsen Zhu](mailto:zhujingsen.p32@gmail.com) to help with potential recovery.

Thank you for your understanding and continued interest in our work.

# InteriorVerse: Large-scale Photorealistic Indoor Scene Dataset

### [Download](Interiorverse_dataset.csv) | [Paper](https://arxiv.org/abs/2211.03017)

![](assets/teaser.gif)

InteriorVerse is a large-scale photorealistic indoor scene dataset proposed by paper [Learning-based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing](https://jingsenzhu.github.io/invrend) in SIGGRAPH Asia'22 conference proceedings. It contains synthetic rendering results of over 4000 indoor scenes with ground truth material, geometry and spatially-varying lightings. **You can find the download links in this [CSV file](Interiorverse_dataset.csv).**

## Material-Geometry Dataset

Our material-geometry dataset contains synthetic ground truths of 50000+ images from 4000+ well-designed indoor scenes.

The dataset folder is organized as in the following structure:

```
./
    dataset_85/
        scene_name_000/
            000_im.exr
            000_mask.exr
            000_albedo.exr
            000_depth.exr
            000_material.exr
            000_normal.exr
            001_im.exr
            ...
        scene_name_001/
            ...
        ...
        train.txt
        val.txt
        test.txt
    dataset_120/
        ... 
        (The same as dataset_85)
```

We provide 2 datasets of different camera intrinsics, which differs in camera horizontal field-of-view angle (85° and 120° respectively), in `dataset_85` and `dataset_120` folder respectively. The camera positions and poses of the same image in 2 datasets are exactly the same. Each folder can be used as an independent dataset, or of course you can combine them together.

- `<image_id>_im.exr` scene image with HDR radiance. If you want to use them in LDR color space, please clip their values to `(0,1)` and perform gamma corrections.
- `<image_id>_mask.exr`: single-channel mask image. 1 means valid pixel while 0 means invalid pixels.
- `<image_id>_albedo.exr`: albedo image.
- `<image_id>_material.exr`: material image. The R channel refers to roughness and G channel refers to metallic, while B channel is unused. Roughness and metallic are defined under GGX microfacet BRDF distribution.
- `<image_id>_depth.exr`: depth image. The unit length is millimeter (mm) and you can normalize them. Note that it contains `inf` values in invalid pixels, please mask them out first.
- `<image_id>_normal.exr`: normal image defined under **OpenGL coordinate system**, i.e. `R` channel (`x` axis) denotes right, `G` channel (`y` axis) denotes up, `B` channel (`z` axis) denotes back (screen to eye).

You can refer to [Our Code](https://github.com/jingsenzhu/IndoorInverseRendering/blob/main/mgnet/dataLoader/__init__.py) for an example data loading process of our dataset.

- About `.exr` format. We suggest using OpenCV to load an `.exr` format HDR image:

  ```python
  import os
  os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1' # Enable OpenCV support for EXR
  import cv2
  ...
  im = cv2.imread(im_path, -1) # im will be an numpy.float32 array of shape (H, W, C)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # cv2 reads image in BGR shape, convert into RGB
  im_ldr = im.clip(0,1)**(1/2.2) # Convert from HDR to LDR with clipping and gamma correction
  ...
  ```

## Lighting Dataset

Coming soon!

## Citation

If you find our work is useful, please consider cite:

```
@inproceedings{zhu2022learning,
    author = {Zhu, Jingsen and Luan, Fujun and Huo, Yuchi and Lin, Zihao and Zhong, Zhihua and Xi, Dianbing and Wang, Rui and Bao, Hujun and Zheng, Jiaxiang and Tang, Rui},
    title = {Learning-Based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing},
    year = {2022},
    publisher = {ACM},
    url = {https://doi.org/10.1145/3550469.3555407},
    booktitle = {SIGGRAPH Asia 2022 Conference Papers},
    articleno = {6},
    numpages = {8}
}
```

