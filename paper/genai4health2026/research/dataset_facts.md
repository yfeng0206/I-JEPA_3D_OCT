# Dataset Facts

## Harvard-FairVision

- [VERIFIED] Harvard-FairVision contains **30,000 subjects total**, comprising separate cohorts of **10,000 subjects each** for age-related macular degeneration, diabetic retinopathy, and glaucoma. Sources: [FairVision dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision/blob/main/README.md); [FairVision paper](https://arxiv.org/abs/2310.02492) [@luo2024fairvision].

- [VERIFIED] The local paper uses the **10,000-subject glaucoma cohort**, not all 30,000 FairVision subjects. Source: [FairVision dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision/blob/main/README.md).

- [VERIFIED] The released glaucoma split contains **6,000 training, 1,000 validation, and 3,000 test subjects**. Sources: FairVision release split metadata and the dataset structure described by the [official repository](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision).

- [VERIFIED] Every FairVision disease cohort contains paired **scanning laser ophthalmoscopy (SLO)** and **3D OCT** data. For glaucoma, the `oct_bscans` array has shape **200 × 200 × 200**. AMD and DR use 128 × 200 × 200 arrays. Source: [FairVision dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision/blob/main/README.md).

- [VERIFIED] A FairVision glaucoma input should be described as a **3D OCT volume or stack of B-scans**, not as “one 3D B-scan.” A B-scan is an individual 2D cross-section; `oct_bscans` stores the volume. Source: [FairVision dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision/blob/main/README.md).

- [VERIFIED] Available demographic attributes are **age, gender/sex, race, ethnicity, preferred language, and marital status**. The release encodes race as Asian, Black, or White; ethnicity as Hispanic or non-Hispanic; and gender as female or male. Sources: [official FairVision repository](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision); [FairVision dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision/blob/main/README.md).

- [VERIFIED] FairVision's **data license is CC BY-NC-ND 4.0**. The data are restricted to non-commercial research and must not be used for clinical decisions or patient care. Source: [FairVision dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision/blob/main/README.md).

- [VERIFIED] The repository's MIT software license does **not** replace or relax the dataset's CC BY-NC-ND 4.0 license. Sources: [official FairVision repository](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision); [FairVision dataset card](https://huggingface.co/datasets/harvardairobotics/FairVision/blob/main/README.md).

- [VERIFIED] The supervised 3D ResNet FairVision baseline reports glaucoma overall AUC **0.8649**, making a frozen-probe AUC near 0.87 comparable to that dataset-specific baseline. This is not a claim of universal OCT state of the art. Source: [FairVision paper, Section VI](https://arxiv.org/html/2310.02492v3) [@luo2024fairvision].

## Harvard-GF3300 — distinct dataset

- [VERIFIED] **Harvard-FairVision and Harvard-GF are distinct datasets and must not be used as interchangeable names.** FairVision contains 30,000 subjects across three diseases; Harvard-GF contains 3,300 glaucoma patients. Sources: [FairVision repository](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision); [Harvard-GF repository](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF); [@luo2024harvardgf].

- [VERIFIED] Harvard-GF contains **3,300 patients**, split into **2,100 training, 300 validation, and 900 test samples**, with equal numbers of Asian, Black, and White patients. Source: [Harvard-GF repository](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF).

- [VERIFIED] Harvard-GF supplies a 200 × 200 RNFL thickness map, a **200 × 200 × 200 OCT volume**, glaucoma status, visual-field mean deviation and total-deviation values, and demographic attributes. Source: [Harvard-GF repository](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF).

- [VERIFIED] Harvard-GF data also use **CC BY-NC-ND 4.0**, are restricted to non-commercial research, and are not licensed for clinical decisions or patient care. Source: [Harvard-GF repository](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF).

## GOALS

- [VERIFIED] GOALS stands for **Glaucoma OCT Analysis and Layer Segmentation** and was released for the MICCAI 2022 challenge. Source: [GOALS paper](https://doi.org/10.1007/978-3-031-16525-2_14) [@fang2022goals].

- [VERIFIED] GOALS contains **300 circumpapillary 2D OCT images** from **99 eyes of 66 people**. It is not a 300-volume glaucoma-classification dataset. Source: [GOALS paper](https://arxiv.org/abs/2207.14447) [@fang2022goals].

- [VERIFIED] Images were acquired using a **TOPCON DRI swept-source OCT** system and distributed as standardized **1100 × 800 PNG images**. Source: [GOALS paper](https://arxiv.org/abs/2207.14447) [@fang2022goals].

- [VERIFIED] GOALS uses a patient-wise **100/100/100 split** for training, preliminary testing, and final testing. Source: [GOALS paper](https://doi.org/10.1007/978-3-031-16525-2_14) [@fang2022goals].

- [VERIFIED] Public ground truth is supplied for the **training set**; preliminary- and final-test labels are withheld for challenge evaluation. Source: [GOALS challenge description](https://aistudio.baidu.com/aistudio/competition/detail/230); [@fang2022goals].

- [VERIFIED] The segmentation classes comprise **retinal nerve fiber layer (RNFL), ganglion cell–inner plexiform layer (GCIPL), choroid, and other/background tissue**. Source: [GOALS paper](https://arxiv.org/abs/2207.14447) [@fang2022goals].

- [UNVERIFIED] No explicit, authoritative legal license for redistribution or derivative release of the GOALS images was recovered. Do not assign GOALS a Creative Commons license without confirmation from the organizers. Source checked: GOALS paper, challenge page, and public release materials.
