# MoocRadar

MoocRadar is maintained by the Knowledge Engineering Group of Tsinghua University with the assistance of Insititute of Education, Tsinghua Univerisity. This repository consists of 2,513 exercises, 14,226 students and over 12 million behavioral data and 5,600 fine-grained concepts, for supporting the developments of cognitive student modeling in MOOCs. The raw data is from XuetangX (https://www.xuetangx.com/).

We summarize the features of MoocRadar as:

* Abundant Learning Context: MoocRadar provides the relevant learning resources, structures, and contents about the students' exercise behaviors, which can enrich the selection candidates for the modeling methods.
* Fine-grained Knowledge Concepts: All the fine-grained concepts have been manually annotated and checked by the experts, which guarantees the quality of such specifical knowledge.
* Cognitive Level Labels: We invoke the _Bloom Cognitive Taxonomy_ to construct "Cognitive Level" tags for the exercises, which can be further explored in subsequent research.

We are still going on the extension and annotation of this repository.

Based on MoocRadar, developers can attempt to build a more informative profile for each student, as introduced in our paper.

![task](https://cloud.tsinghua.edu.cn/f/75a1dcfb41a84b7aaeb0/?dl=1)

## News !! 

* Exercise amount is extended to 9,384 !!

* Our paper is submitted to SIGIR resource track !!
* Update the annotation guidance of fine-grained concepts and cognitive labels.

## Data Access

There are multi-level data to be used, including:

| Dataset          | Description                                           | Download Link |
| ---------------- | ----------------------------------------------------- | ------------- |
| MoocRadar_Raw    | The raw data from MOOCCubeX (after data filtering).   |     [Raw link](https://cloud.tsinghua.edu.cn/d/adc2d43d154944ffb75f/)          |
| MoocRadar_Coarse | Exercises and behaviors with coarse-grained concepts. |      [Coarse link](https://cloud.tsinghua.edu.cn/d/5443ee05152344c79419/)         |
| MoocRadar_Middle | Exercises and behaviors with middle-grained concepts. |     [Middle link](https://cloud.tsinghua.edu.cn/d/adf72390e3234143aec0/)          |
| MoocRadar_Fine   | Exercises and behaviors with fine-grained concepts.   |      [Fine link](https://cloud.tsinghua.edu.cn/d/308c17eeb99e4ebf98e2/)         |
| External_Data    | Other additional data of MoocRadar.                   |     [External link](https://cloud.tsinghua.edu.cn/d/000fddd19a434765872a/)          |

## Reproduction Model

Rsearchers can set up the presented models with [EduKTM](https://github.com/bigdata-ustc/EduKTM) and [EduCDM](https://github.com/bigdata-ustc/EduCDM).

We provide several basic model's demo, including:

* Knowledge Tracing:
  * [DKT](./baselines/scripts/DKT.sh)
  * [DKT+](./baselines/scripts/DKTplus.sh)
  * [DKVMN](./baselines/scripts/DKVMN.sh)

* Cognitive Diagnosis:
  * [GDIRT](./baselines/scripts/GDIRT.sh)
  * [MIRT](./baselines/scripts/MIRT.sh)
  * [NCDM](./baselines/scripts/NCDM.sh)

We also provide the performance of the improvement of DKVMN and NCDM with side information (i.e. cognitive and video).

* +Cognitive:
  * [DKVMN](./baselines-cognitive/scripts/DKVMN.sh)
  * [NCDM](./baselines-cognitive/scripts/NCDM.sh)

* +Video:
  * [DKVMN](./baselines-video/scripts/DKVMN.sh)
  * [NCDM](./baselines-video/scripts/NCDM.sh)

### Data for baselines reproduction:

1. `--mode` (Option: Coarse/Middle/Fine) for your settings
2. `--data_dir` with Corresponding granularity data from above table.
    
    for example, for `--mode Middle` setting, prepare the following files:
    - `./data/student-problem-middle.json`
    - `./data/problem.json`
3. then generate train/test dataset by setting: `--data_process` in scripts

### Data for improvement reproduction with cognitive and video side information:

Option 1: generate by setting: `--data_process` in scripts

Option 2: download from [there](https://cloud.tsinghua.edu.cn/d/b95c63db77c84657a8a4/)

## Toolkit & Guidance

There are also several tools and guidance for extending and employing the data.

For extending the data from MOOCCubeX knowledge base.

* Step 1: Download raw data from https://github.com/THU-KEG/MOOCCubeX.
* Step 2: Build the concepts with MOOCCube Cocnept Helper.

For further data annotation:

* The annotation guidance for fine-grained concepts and cognitive labels
* (Currently Chinese Only) https://cloud.tsinghua.edu.cn/f/cdfdeca0893e4ed0ac9c/

For more information:

* About learning sequence representation: 
  * https://arxiv.org/pdf/2208.04708
* About Bloom Cognitive Taxonomy:
  * http://www.edpsycinteractive.org/topics/cognition/bloom.pdf
* About the Fine-grained Concept Extraction:
  * Ver 0.5: https://github.com/yujifan0326/Concept-Acquisition-Pipeline
  * Ver 1.0: On the development.

## Feature

The distribution of students' exercise behaviors, accurate rates and concept-linked exercises.

![](https://cloud.tsinghua.edu.cn/f/2daf2a6ddb4d497b97c9/?dl=1)

## Reference

 @article{MOOCRadar,
    title={MoocRadar: A Fine-grained and Multi-aspect Knowledge Repository for Improving Cognitive Student Modeling in MOOCs},
    author={Jifan Yu, Mengying Lu, Qingyang Zhong, Zijun Yao, Shangqing Tu, Zhengshan Liao, Xiaoya Li, Manli Li, Lei Hou, Haitao Zheng, Juanzi Li, Jie Tang},
    year={ 2023 }
    }
