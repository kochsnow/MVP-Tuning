# coding=utf-8

import ast
import csv
import os
import json
import pdb
import datasets
import random


_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = "CC-BY-SA-4.0 License"




_TEST_FILE = "./data/obqa/statement/test.fact_mvp_statement.jsonl"
_TRAINING_FILE = "./data/obqa/statement/train.fact_mvp_statement.jsonl"
_DEV_FILE = "./data/obqa/statement/dev.fact_mvp_statement.jsonl"


label2id = {'A':0, 'B':1, 'C':2, 'D':3}#, 'E':4}

class OBQA(datasets.GeneratorBasedBuilder):
    """The FETAQA dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question_choice_A": datasets.Value("string"),
                    "question_choice_B": datasets.Value("string"),
                    "question_choice_C": datasets.Value("string"),
                    "question_choice_D": datasets.Value("string"),
                    # "question_choice_E": datasets.Value("string"),
                    "triplets_a": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    "triplets_b": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    "triplets_c": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    "triplets_d": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    # "triplets_e": {"individual_view_triplets": datasets.Value("string"), 
                    #             "latent_view_triplets":  datasets.Value("string"),
                    #             "group_view_triplets": datasets.Value("string"),
                    #             "retri_view_triplets": datasets.Value("string"),
                    #             "selected_individual_view_triplets": datasets.Value("string"),
                    #             "meaning_view_triplets": datasets.Value("string")},
                    "labels": datasets.Value("int32"),
                    # "question_concept": datasets.Value("string")
                } 
            ),
            supervised_keys=None,
            homepage=None,
            license=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
          datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": _TRAINING_FILE}),
          datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": _DEV_FILE}),
          datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": _TEST_FILE}),
        #   datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            examples = json.load(f)
            for idx, example in enumerate(examples):            
                yield idx, {
                    "id": example["id"],
                    "question_choice_A": example['question']['stem'] + ' ' + example['question']['choices'][0]['text'] + ' ' + example['fact1'],
                    "question_choice_B": example['question']['stem'] + ' ' + example['question']['choices'][1]['text'] + ' ' + example['fact1'],
                    "question_choice_C": example['question']['stem'] + ' ' + example['question']['choices'][2]['text'] + ' ' + example['fact1'],
                    "question_choice_D": example['question']['stem'] + ' ' + example['question']['choices'][3]['text'] + ' ' + example['fact1'],
                    "labels": label2id[example["answerKey"]],
                    "triplets_a": {'individual_view_triplets': example["statements"][0]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['question']['choices'][0]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][0]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                    "triplets_b": {'individual_view_triplets': example["statements"][1]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['question']['choices'][1]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][1]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                    "triplets_c": {'individual_view_triplets': example["statements"][2]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['question']['choices'][2]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][2]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                    "triplets_d": {'individual_view_triplets': example["statements"][3]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['question']['choices'][3]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][3]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                }
