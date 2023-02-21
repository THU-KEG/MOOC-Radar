# MoocRadar

MoocRadar is maintained by the Knowledge Engineering Group of Tsinghua University with the assistance of Insititute of Education, Tsinghua Univerisity. This repository consists of 2,513 exercises, 14,226 students and over 12 million behavioral data and 5,600 fine-grained concepts, for supporting the developments of cognitive student modeling in MOOCs.

We summarize the features of MoocRadar as:

* Abundant Learning Context: MoocRadar provides the relevant learning resources, structures, and contents about the students' exercise behaviors, which can enrich the selection candidates for the modeling methods.
* Fine-grained Knowledge Concepts: All the fine-grained concepts have been manually annotated and checked by the experts, which guarantees the quality of such specifical knowledge.
* Cognitive Level Labels: We invoke the _Bloom Cognitive Taxonomy_ to construct "Cognitive Level" tags for the exercises, which can be further explored in subsequent research.

We are still going on the extension and annotation of this repository.

Based on MoocRadar, developers can attempt to build a more informative profile for each student, as introduced in our paper.

![task](https://cloud.tsinghua.edu.cn/f/ca31a00f0131409089fe/?dl=1)

## News !! 

* Exercise amount is extended to 9,384 !!

* Our paper is submitted to SIGIR resource track !!

## Data Access

There are multi-level data to be used, including:

| Dataset          | Description                                           | Download Link |
| ---------------- | ----------------------------------------------------- | ------------- |
| MoocRadar_Raw    | The raw data from MOOCCubeX (after data filtering).   |               |
| MoocRadar_Coarse | Exercises and behaviors with coarse-grained concepts. |               |
| MoocRadar_Middle | Exercises and behaviors with middle-grained concepts. |               |
| MoocRadar_Fine   | Exercises and behaviors with fine-grained concepts.   |               |
| External_Data    | Other additional data of MoocRadar.                   |               |

## Reproduction Model

Rsearchers can set up the presented models with [EduKTM](https://github.com/bigdata-ustc/EduKTM) and [EduCDM](https://github.com/bigdata-ustc/EduCDM).

We provide several basic model's demo, including:

* Knowledge Tracing:
  * DKT (CodeLink), DKT+ (CodeLink), DKVMN (CodeLink)

* Cognitive Diagnosis:
  * GDIRT (CodeLink), MIRT (CodeLink), NCDM (CodeLink)

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

To be updated.