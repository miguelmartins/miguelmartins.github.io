---
title: "CircorDigiscope dataset now available at HuggingFace dataset hub!"
date: 2025-07-03
permalink: /posts/2025/07/circor-digiscope-huggingface/
tags:
  - cool posts
  - category1
  - category2
---

<!-- In my first and (to date) only job has a Data Scientist I remember discovering this new website, HuggingFace. Has newby to the deep learning scene, I was fascinated the ease to which I could use state-of-the-art models that were already setup for me. All of this, for free! -->
<!---->
<!-- As I then moved to academia I did found myself implementing most of the models from scratch whenever I could. This would prove a valuable exercise, of course. Notwithstanding, whenever I had to build any experimental benchmark, I still had to go through the painfull process of processing datasets, that more often than not come in a very heterogenous (and sometimes archaic way). -->

My first task as a PhD. student was to design a new algorithm for heart sound segmentation. My advisor already had a great idea on how we could potentially advance the state-of-the-art. I got the implementation ready rather quickly and it seem to be working correctly using our numerical tests. There was just one important step missing: _testing with actual heart sound data_.

This was especially important since our lab was involved in collecting a very important new dataset for this purpose, the [CirCor DigiScope Dataset](https://physionet.org/content/circor-heart-sound/1.0.3/). It still is the biggest public dataset with recordings of **pediatric** heart sounds.

There was just a problem: processing the data was a nightmare. The documentation was quite lengthy, the annotations a bit confusing -- the fruits of my labour can still be found on my [github](https://github.com/miguelmartins/mnn/blob/main/data_processing/signal_extraction.py). Regardless of the amateurish academic code, I thought making it public was already a mighty effort, since it was also mentioned in [our journal publication](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10242001).

How naive I was as a first year PhD. student. My code required a bunch of dependecies, clearly being very far from a decent plug and play solution. In the best case, only people belonging to the very niche research community of heart sounds would benefit from this implementation.

I stumbled upon a solution to this problem after attending a talk by [Nouamane Tazi](https://www.linkedin.com/in/nouamanetazi/) from [HuggingFace](https://huggingface.co/) at my University. Nouamane was working in foundational models and pointed out that making these data more accessible would probably benefit folks working on training and developing big foundational models.

# All fine and dandy. Show me how it works!

Two versions of the same data are available: one with the heart sounds and labels in a raw format, another with our own pre-processing functions. The pre-processed version is heavily inspired on the [Matlab code of Springer et al.](https://github.com/davidspringer/Springer-Segmentation-Code), but it is based on [our own python implementation](https://github.com/miguelmartins/mnn/blob/main/data_processing/signal_extraction.py).

The dataset can be viewed and interacted with on my [HuggingFace repo here](https://huggingface.co/datasets/miguellmartins/circor-digiscope-physionet22-processed).

I will focus on just loading the data so that you can use it for your own use-cases. If you are interested in a heart sound segmentation example, [you can follow our U-Net segmentation tutorial here](/talks/2025-03-08-bip-tutorial).

After installing [HuggingFace datasets for python](https://huggingface.co/docs/datasets/en/installation), simply run:

```python
from datasets import load_dataset, Audio, DatasetDic
circor = load_dataset('miguellmartins/circor-digiscope-physionet22-processed')

print(circor)
```

The `load_dataset` method will download the files stored remotely on HuggingFace hub, and it will return a `DatasetDict` containing two splits: the _original_ dataset, i.e., the raw sounds without any sort of preprocessing of filter extraction, and the _processed_ split, with specialized filtering, downsampling and feature extraction.

```console
DatasetDict({
    original: Dataset({
    features: ['filename', 'recording', 'recording_label', 'heart_state_labels'],
    num_rows: 3363
    })
    processed: Dataset({
    features: ['filename', 'recording', 'recording_label', 'heart_state_labels'],
    num_rows: 3363
    })
})

```

Let us focus on the unprocessed files since this post is targeted at a general audience. You can access each dataset by its key on the `DatasetDict`

```python
original = circor['original']
print(original)
```

and we get the set of attributes:

```console
Dataset({
    features: ['filename', 'recording', 'recording_label', 'heart_state_labels'],
    num_rows: 3363
})
```

Each feature has the following attributes:

- `recording` the waveform of the recording plus recording metadata
- `recording_label` the global label of the recording, can be 0 (normal) or 1 (abnormal)
- `heart_state_labels` annotations of the heart state for each sample

One can access the dataset by index, and each of these attributes by name. For instance:

```python
print(original[0]['recording'])
```

Accesses the recording attribute of the first observation in our dataset. Running the above code yields the following output:

```console
{'path': '13918_AV.wav',
 'array': array([-0.0100708 , -0.00579834, -0.00692749, ..., -0.00238037,
         0.00396729,  0.00717163]),
 'sampling_rate': 4000}
```

The actual waveform can be acessed using the key `array`, and `sampling_rate` yields the acquisition sampling rate in Hertz.

`path` is a string of the type `{PID}_{LOCATION}.{EXTENSION}`:

- `PID` is the patient identifier, this is crucial if you do not want to have patien-wise leakage between your train and validation splits.
- `LOCATION` tells you the anatomical screening location: it can be PV, TV, AV, MV, or Phc.
- `EXTENSION` tell you the file extension where the original recording was stored.

Note that by this stage you are now ready to use the recording data to train your models!

For instance, if you are using [PyTorch](https://pytorch.org/), the simplest way to go about it is:

```python
ds = original.with_format("torch", device=your_device)
# your pre-preprocessing here...

# model training and validation here...

# ?????

# profit
```

This about wraps it up from a bird's eye perpesctive. I do encourage you to into look the [original documentation](https://huggingface.co/docs/datasets/en/package_reference/main_classes) to understand the full picture.

Feel free to reach out with any questions. Note that we only processed the data directly related with heart sound segmentation. If you require the entirity of the metadata per patient, please refer to
[CirCor DigiScope Dataset's original website](https://physionet.org/content/circor-heart-sound/1.0.3/).
