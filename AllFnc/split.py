from decimal import Decimal, ROUND_HALF_UP
import math
import random

__all__ = ['create_nested_kfold_subject_split',
           'merge_partition_lists',
           'merge_partition_lists_noshift',
          ]

def create_nested_kfold_subject_split(subj: int or list[int], 
                                      folds: int = 10, 
                                      folds_nested: int = 5, 
                                      start_with: int = 1,
                                      seed: int = 83136297, 
                                     ) -> list[list[int], list[int], list[int]]:
    '''
    create_nested_kfold_subject_split creates a partition list to run a
    "nested subject-based k-fold cross validation". 

    Parameters
    ----------
    subj: int or list[int]
        The number of subjects of the dataset. It can be a list with the specific
        subject ids to use during the data partition.
    folds: int, optional
        The number of outer folds. It must be a positive integer. 
        Default = 10
    folds_nested: int, optional
        The number of inner folds. It must be a positive integer with value
        bigger than floor(subj-subj/folds).
        Default = 5
    start_with: int, optional
        The starting ID to use when creating the list of subject IDs. It will
        be considered only if subj is given as an integer. For example, if subj 
        is 10 and start_with is 5, the list of IDs to partition is 
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14].
        Default = 1
    seed: int, optional
        The seed to use during data partition. This will make results reproducible.
        Default = 83136297

    Returns
    -------
    partition_list: list
        The partition list. Each element in the list is a collection 
        of three lists, representing a specific partition. 
        The first contains the subjects IDs included in the training set,
        the second the subjects IDs included in the validation set,
        the third the subjects IDs included in the test set.
        Note that the length of the partition_list is outer_folds*inner_folds,
        and the partition at index i refers to the 
        mod(i+1, inner_folds) inner fold of the ceil((i+1)/outer_folds) outer fold.
    
    '''
    # initialize list of subjects
    if isinstance(subj,list):
        subj_list = subj
        subj = len(subj_list)
    else:
        start_idx = start_with
        end_idx = start_idx + subj
        subj_list = [i for i in range(start_idx, end_idx)]
    
    if folds>subj:
        raise ValueError('Number of folds cannot be greater' +
                         ' than the number of subjects')
    if folds_nested > math.floor(subj-subj/folds):
        raise ValueError(
            '''
            Number of nested folds cannot be greater than the number of subjects 
            minus the number of subject in the test fold, so you must satisfy 
            ceil(subjects - subj/folds) > folds_nested
            ''' 
        )

    # to calculate list diffs by preserving order we will use this lambda function
    list_diff = lambda l1,l2: [x for x in l1 if x not in l2]
    
    # preallocate list of split. Each split is represented by three list:
    # the first for train, the second for validation, the third for test
    partition_list = [[[None],[None],[None]] for i in range(folds*folds_nested)]

    # shuffle list according to given seed
    random.seed(seed)
    random.shuffle(subj_list)
    subj_list_set = set(subj_list)
    
    # initialize test counter to provide a simple way 
    # to create splits with a similar size
    prt_cnt = 0
    tst_cnt = 0
    for i in range(folds):
        
        # each nested fold share the same test set so we initialize it outside the 
        # second loop the best way to create test sets is to slide along the 
        # shuffled list, then get the remaining subject for the next for loop
        tst_to_add = Decimal((subj-tst_cnt)/(folds-i))
        tst_to_add = int(tst_to_add.to_integral_value(rounding=ROUND_HALF_UP))
        tst_id = subj_list[tst_cnt:(tst_cnt+tst_to_add)]
        tst_cnt += tst_to_add

        # extract list of remaining subjects for the creation of the train/val split
        val_cnt = 0
        subj_left = list_diff(subj_list,tst_id)
        subj_left_set = set(subj_left)
        Nleft = len(subj_left)
        for k in range(folds_nested):

            # we need to create a train and validation from the remaining subjects 
            # the best way to do it is to create the validation by sliding along 
            # the list of remaining subjects
            val_to_add = Decimal((Nleft-val_cnt)/(folds_nested-k))
            val_to_add = int(val_to_add.to_integral_value(rounding=ROUND_HALF_UP))
            val_id = subj_left[val_cnt:(val_cnt+val_to_add)]
            val_cnt += val_to_add

            # finally we create the last
            trn_id = subj_left_set.difference(set(val_id))

            #assign everything
            partition_list[prt_cnt][0]=list(trn_id)
            partition_list[prt_cnt][1]=list(val_id)
            partition_list[prt_cnt][2]=list(tst_id)
            prt_cnt += 1

    return partition_list


def merge_partition_lists_noshift(l1: list[list[int], list[int], list[int]], 
                                  l2: list[list[int], list[int], list[int]]
                                 ) -> list[list[int], list[int], list[int]]:
    '''
    merge_partition_lists_noshift merges the rows of two partition list with the
    same length. Partitions are supposed to come from the 
    create_nested_kfold_subject_split function.

    Parameters
    ----------
    l1: list
        The first partition list. It must be a list of three integer lists,
        the first with the subject's training IDs, the second with the subject's
        validation IDs, and the third with the subject's test IDs. For example,
        [ [[1,2,3], [4,5] ,[6,7]], [[4,5],[1,2,3],[6,7]] ]
    l2: list
        The second partition list. Same as l1. The length of l1 must be equal 
        to the length of l2

    Returns
    -------
    new_list: list
        The merged partition list. The structure is the same as l1 and l2.
    
    '''
    
    if len(l1) != len(l2):
        ValueError('the two partition lists must be of the same length')

    new_list = [[[None],[None],[None]] for i in range(len(l1))]
    for i in range(len(l1)):
        new_list[i][0] = list(set(l1[i][0] + l2[i][0]))
        new_list[i][1] = list(set(l1[i][1] + l2[i][1]))
        new_list[i][2] = list(set(l1[i][2] + l2[i][2]))

    return new_list


def merge_partition_lists(l1: list[list[int], list[int], list[int]], 
                          l2: list[list[int], list[int], list[int]],
                          folds: int,
                          folds_nested: int
                         ) -> list[list[int], list[int], list[int]]:
    '''
    merge_partition_lists merges the rows of two partition list with the
    same length. Partitions are supposed to come from the 
    create_nested_kfold_subject_split function.
    To avoid the creation of sets with too different size
    l2 rows will be selected by shifting 1 folds + 1 nested folds. 
    For example, in case of a 10 fold CV with 5 nested folds, the following 
    couples of rows will be selected:
    
        [0 6],  [1 7],  [2 8],  [3 9],  [4 5],
        [5 11], [6 12], [7 13], [8 14], [9 10],
        [10 16],[11 17],[12 18],[13 19],[14 15],
        [15 21],[16 22],[17 23],[18 24],[19 20],
        [20 26],[21 27],[22 28],[23 29],[24 25],
        [25 31],[26 32],[27 33],[28 34],[29 30],
        [30 36],[31 37],[32 38],[33 39],[34 35],
        [35 41],[36 42],[37 43],[38 44],[39 40],
        [40 46],[41 47],[42 48],[43 49],[44 45],
        [45 1], [46 2], [47 3], [48 4], [49 0],

    Parameters
    ----------
    l1: list
        The first partition list. It must be a list of three integer lists,
        the first with the subject's training IDs, the second with the subject's
        validation IDs, and the third with the subject's test IDs. For example,
        [ [[1,2,3], [4,5] ,[6,7]], [[4,5],[1,2,3],[6,7]] ]
    l2: list
        The second partition list. Same as l1. The length of l1 must be equal 
        to the length of l2
    folds: int
        The number of outer folds. It must be the same parameter given to the 
        create_nested_kfold_subject_split function.
    folds_nested: int
        The number of nested folds. It must be the same parameter given to the 
        create_nested_kfold_subject_split function.

    Returns
    -------
    new_list: list
        The merged partition list with fusion shift paradigm. 
        The structure is the same as l1 and l2.
        
    '''
    if len(l1) != len(l2):
        ValueError('the two partition lists must be of the same length')

    new_list = [[[None],[None],[None]] for i in range(len(l1))]
    for i in range(folds):
        for k in range(folds_nested):
            l1_idx = folds_nested*i+k
            if k == folds_nested-1:
                l2_idx = folds_nested*(i+1)
            else: 
                l2_idx = folds_nested*(i+1) + k + 1
            if i == folds - 1:
                l2_idx = 0 if k == folds_nested-1 else k + 1
            new_list[l1_idx][0] = list(set(l1[l1_idx][0] + l2[l2_idx][0]))
            new_list[l1_idx][1] = list(set(l1[l1_idx][1] + l2[l2_idx][1]))
            new_list[l1_idx][2] = list(set(l1[l1_idx][2] + l2[l2_idx][2]))

    return new_list

