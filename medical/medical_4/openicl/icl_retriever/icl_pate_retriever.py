"""Random Retriever"""

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)


class PateRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class
        Class of Random Retriever.

    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        seed (`int`, optional): Seed for the random number generator.
    """

    def __init__(
        self,
        dataset_reader: DatasetReader,
        ice_separator: Optional[str] = "\n",
        ice_eos_token: Optional[str] = "\n",
        prompt_eos_token: Optional[str] = "",
        ice_num: Optional[int] = 1,
        index_split: Optional[str] = "train",
        test_split: Optional[str] = "test",
        seed: Optional[int] = 43,
        accelerator: Optional[Accelerator] = None,
        ensemble: Optional[int] = 1,
    ) -> None:
        super().__init__(
            dataset_reader,
            ice_separator,
            ice_eos_token,
            prompt_eos_token,
            ice_num,
            index_split,
            test_split,
            accelerator,
        )
        self.seed = seed
        self.ensemble = ensemble

    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            idx_list = np.random.choice(
                num_idx, self.ice_num * self.ensemble, replace=False
            ).tolist()
            rtr_idx_list.append(idx_list)
        all_ensemble_idx_list = []
        for i in range(self.ensemble):
            idx_list_temp = []
            for j in range(len(rtr_idx_list)):
                idx_list_temp.append(
                    rtr_idx_list[j][i * self.ice_num : (i + 1) * self.ice_num]
                )
            all_ensemble_idx_list.append(idx_list_temp)
        return all_ensemble_idx_list

    # def retrieve(self):
    #     np.random.seed(self.seed)
    #     num_idx = len(self.index_ds)
    #     rtr_idx_list = []
    #     logger.info("Retrieving data for test set...")
    #     for _ in trange(len(self.test_ds), disable=not self.is_main_process):
    #         idx_list = np.random.choice(num_idx, self.ice_num*self.ensemble*2, replace=False).tolist()
    #         rtr_idx_list.append(idx_list)
    #     all_ensemble_idx_list = []
    #     for i in range(self.ensemble):
    #         idx_list_temp = []
    #         for j in range(len(rtr_idx_list)):
    #             idx_list_temp.append(rtr_idx_list[j][200+i*self.ice_num:200+(i+1)*self.ice_num])
    #         all_ensemble_idx_list.append(idx_list_temp)
    #     # print(len(all_ensemble_idx_list),len(all_ensemble_idx_list[0]),all_ensemble_idx_list[0][0],all_ensemble_idx_list[1][0],
    #     #     all_ensemble_idx_list[2][0],all_ensemble_idx_list[3][0],all_ensemble_idx_list[4][0],all_ensemble_idx_list[5][0])
    #     return all_ensemble_idx_list
