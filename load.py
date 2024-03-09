import pickle

from utils.loader import *
from utils.utility import *

def load_dataset(args):
    DATA = args.data
    LOAD = args.LOAD
    MODEL = args.model
    FEATURES = args.features
    LABELS = args.labels
    
    if DATA == 'adni':
        ### Select the feature files and labels
        features_list, labels_dict = set_classification_conditions(args)
        
        if LOAD:
            if LABELS == 0:
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/5way/data_set_0_svm.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/5way/data_set_1_svm.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/5way/data_set_2_svm.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/5way/data_set_3_svm.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/5way/data_set_4_svm.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/5way/data_set_5_svm.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/5way/data_set_6_svm.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/5way/data_set_7_svm.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/5way/data_set_0_gdc.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/5way/data_set_1_gdc.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/5way/data_set_2_gdc.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/5way/data_set_3_gdc.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/5way/data_set_4_gdc.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/5way/data_set_5_gdc.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/5way/data_set_6_gdc.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/5way/data_set_7_gdc.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                else:
                    if FEATURES == 0:
                        with open('variables/5way/data_set_0.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/5way/data_set_1.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/5way/data_set_2.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/5way/data_set_3.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/5way/data_set_4.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/5way/data_set_5.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/5way/data_set_6.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/5way/data_set_7.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
            elif LABELS == 1: # CN + SMC vs EMCI + LMCI vs AD
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/3way/data_set_0_svm.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/3way/data_set_1_svm.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/3way/data_set_2_svm.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/3way/data_set_3_svm.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/3way/data_set_4_svm.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/3way/data_set_5_svm.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/3way/data_set_6_svm.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/3way/data_set_7_svm.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/3way/data_set_0_gdc.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/3way/data_set_1_gdc.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/3way/data_set_2_gdc.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/3way/data_set_3_gdc.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/3way/data_set_4_gdc.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/3way/data_set_5_gdc.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/3way/data_set_6_gdc.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/3way/data_set_7_gdc.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                else:
                    if FEATURES == 0:
                        with open('variables/3way/data_set_0.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/3way/data_set_1.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/3way/data_set_2.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/3way/data_set_3.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/3way/data_set_4.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/3way/data_set_5.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/3way/data_set_6.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/3way/data_set_7.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
            elif LABELS == 2: # Pre vs MCI
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/pre_mci/data_set_0_svm.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/pre_mci/data_set_1_svm.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/pre_mci/data_set_2_svm.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/pre_mci/data_set_3_svm.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/pre_mci/data_set_4_svm.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/pre_mci/data_set_5_svm.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/pre_mci/data_set_6_svm.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/pre_mci/data_set_7_svm.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/pre_mci/data_set_0_gdc.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/pre_mci/data_set_1_gdc.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/pre_mci/data_set_2_gdc.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/pre_mci/data_set_3_gdc.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/pre_mci/data_set_4_gdc.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/pre_mci/data_set_5_gdc.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/pre_mci/data_set_6_gdc.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/pre_mci/data_set_7_gdc.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                else:
                    if FEATURES == 0:
                        with open('variables/pre_mci/data_set_0.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/pre_mci/data_set_1.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/pre_mci/data_set_2.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/pre_mci/data_set_3.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/pre_mci/data_set_4.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/pre_mci/data_set_5.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/pre_mci/data_set_6.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/pre_mci/data_set_7.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
            elif LABELS == 3: # Pre vs AD
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/pre_ad/data_set_0_svm.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/pre_ad/data_set_1_svm.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/pre_ad/data_set_2_svm.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/pre_ad/data_set_3_svm.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/pre_ad/data_set_4_svm.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/pre_ad/data_set_5_svm.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/pre_ad/data_set_6_svm.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/pre_ad/data_set_7_svm.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/pre_ad/data_set_0_gdc.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/pre_ad/data_set_1_gdc.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/pre_ad/data_set_2_gdc.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/pre_ad/data_set_3_gdc.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/pre_ad/data_set_4_gdc.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/pre_ad/data_set_5_gdc.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/pre_ad/data_set_6_gdc.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/pre_ad/data_set_7_gdc.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                else:
                    if FEATURES == 0:
                        with open('variables/pre_ad/data_set_0.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/pre_ad/data_set_1.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/pre_ad/data_set_2.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/pre_ad/data_set_3.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/pre_ad/data_set_4.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/pre_ad/data_set_5.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/pre_ad/data_set_6.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/pre_ad/data_set_7.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
            elif LABELS == 4: # MCI vs AD
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0_svm.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1_svm.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2_svm.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3_svm.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/mci_ad/data_set_4_svm.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/mci_ad/data_set_5_svm.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/mci_ad/data_set_6_svm.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/mci_ad/data_set_7_svm.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0_gdc.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1_gdc.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2_gdc.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3_gdc.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/mci_ad/data_set_4_gdc.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/mci_ad/data_set_5_gdc.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/mci_ad/data_set_6_gdc.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/mci_ad/data_set_7_gdc.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                else:
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/mci_ad/data_set_4.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/mci_ad/data_set_5.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/mci_ad/data_set_6.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/mci_ad/data_set_7.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
            elif LABELS == 6: # CN vs SMC vs EMCI
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/3way_pre/data_set_0_svm.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/3way_pre/data_set_1_svm.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/3way_pre/data_set_2_svm.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/3way_pre/data_set_3_svm.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/3way_pre/data_set_4_svm.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/3way_pre/data_set_5_svm.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/3way_pre/data_set_6_svm.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/3way_pre/data_set_7_svm.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0_gdc.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1_gdc.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2_gdc.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3_gdc.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/3way_pre/data_set_4_gdc.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/3way_pre/data_set_5_gdc.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/3way_pre/data_set_6_gdc.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/3way_pre/data_set_7_gdc.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
                else:
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0.pickle', 'rb') as data_set_0:
                            data_set = pickle.load(data_set_0)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1.pickle', 'rb') as data_set_1:
                            data_set = pickle.load(data_set_1)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2.pickle', 'rb') as data_set_2:
                            data_set = pickle.load(data_set_2)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3.pickle', 'rb') as data_set_3:
                            data_set = pickle.load(data_set_3)
                    elif FEATURES == 4:
                        with open('variables/3way_pre/data_set_4.pickle', 'rb') as data_set_4:
                            data_set = pickle.load(data_set_4)
                    elif FEATURES == 5:
                        with open('variables/3way_pre/data_set_5.pickle', 'rb') as data_set_5:
                            data_set = pickle.load(data_set_5)
                    elif FEATURES == 6:
                        with open('variables/3way_pre/data_set_6.pickle', 'rb') as data_set_6:
                            data_set = pickle.load(data_set_6)
                    elif FEATURES == 7:
                        with open('variables/3way_pre/data_set_7.pickle', 'rb') as data_set_7:
                            data_set = pickle.load(data_set_7)
            
            used_features, A, X, y, eigenvalues, eigenvectors, laplacians = data_set
            print(">> Data Load!")
        else:
            ### Get dictionary for subject with ROI features
            features_list, subject_map = get_roi_feature(args, features_list, labels_dict) # {Subject : [ROI features, label]}

            ### Load the graph data
            used_features, A, X, y, eigenvalues, eigenvectors, laplacians = load_graph_data(args, features_list, subject_map)
            
            if LABELS == 0:
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/5way/data_set_0_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/5way/data_set_1_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/5way/data_set_2_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/5way/data_set_3_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/5way/data_set_4_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/5way/data_set_5_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/5way/data_set_6_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/5way/data_set_7_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/5way/data_set_0_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/5way/data_set_1_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/5way/data_set_2_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/5way/data_set_3_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/5way/data_set_4_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/5way/data_set_5_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/5way/data_set_6_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/5way/data_set_7_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                else:
                    if FEATURES == 0:
                        with open('variables/5way/data_set_0.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/5way/data_set_1.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/5way/data_set_2.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/5way/data_set_3.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/5way/data_set_4.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/5way/data_set_5.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/5way/data_set_6.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/5way/data_set_7.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
            elif LABELS == 1: # CN + SMC vs EMCI + LMCI vs AD
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/3way/data_set_0_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/3way/data_set_1_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/3way/data_set_2_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/3way/data_set_3_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/3way/data_set_4_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/3way/data_set_5_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/3way/data_set_6_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/3way/data_set_7_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/3way/data_set_0_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/3way/data_set_1_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/3way/data_set_2_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/3way/data_set_3_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/3way/data_set_4_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/3way/data_set_5_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/3way/data_set_6_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/3way/data_set_7_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                else:
                    if FEATURES == 0:
                        with open('variables/3way/data_set_0.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/3way/data_set_1.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/3way/data_set_2.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/3way/data_set_3.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/3way/data_set_4.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/3way/data_set_5.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/3way/data_set_6.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/3way/data_set_7.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
            elif LABELS == 2: # Pre vs MCI
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/pre_mci/data_set_0_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/pre_mci/data_set_1_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/pre_mci/data_set_2_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/pre_mci/data_set_3_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/pre_mci/data_set_4_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/pre_mci/data_set_5_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/pre_mci/data_set_6_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/pre_mci/data_set_7_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/pre_mci/data_set_0_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/pre_mci/data_set_1_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/pre_mci/data_set_2_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/pre_mci/data_set_3_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/pre_mci/data_set_4_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/pre_mci/data_set_5_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/pre_mci/data_set_6_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/pre_mci/data_set_7_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                else:
                    if FEATURES == 0:
                        with open('variables/pre_mci/data_set_0.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/pre_mci/data_set_1.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/pre_mci/data_set_2.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/pre_mci/data_set_3.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/pre_mci/data_set_4.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/pre_mci/data_set_5.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/pre_mci/data_set_6.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/pre_mci/data_set_7.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
            elif LABELS == 3: # Pre vs AD
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/pre_ad/data_set_0_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/pre_ad/data_set_1_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/pre_ad/data_set_2_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/pre_ad/data_set_3_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/pre_ad/data_set_4_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/pre_ad/data_set_5_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/pre_ad/data_set_6_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/pre_ad/data_set_7_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/pre_ad/data_set_0_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/pre_ad/data_set_1_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/pre_ad/data_set_2_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/pre_ad/data_set_3_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/pre_ad/data_set_4_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/pre_ad/data_set_5_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/pre_ad/data_set_6_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/pre_ad/data_set_7_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                else:
                    if FEATURES == 0:
                        with open('variables/pre_ad/data_set_0.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/pre_ad/data_set_1.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/pre_ad/data_set_2.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/pre_ad/data_set_3.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/pre_ad/data_set_4.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/pre_ad/data_set_5.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/pre_ad/data_set_6.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/pre_ad/data_set_7.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
            elif LABELS == 4: # MCI vs AD
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/mci_ad/data_set_4_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/mci_ad/data_set_5_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/mci_ad/data_set_6_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/mci_ad/data_set_7_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/mci_ad/data_set_4_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/mci_ad/data_set_5_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/mci_ad/data_set_6_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/mci_ad/data_set_7_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                else:
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/mci_ad/data_set_4.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/mci_ad/data_set_5.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/mci_ad/data_set_6.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/mci_ad/data_set_7.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
            elif LABELS == 6: # CN vs SMC vs EMCI
                if MODEL == 'svm':
                    if FEATURES == 0:
                        with open('variables/3way_pre/data_set_0_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/3way_pre/data_set_1_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/3way_pre/data_set_4_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/3way_pre/data_set_5_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/3way_pre/data_set_6_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/3way_pre/data_set_7_svm.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                elif MODEL == 'gdc':
                    if FEATURES == 0:
                        with open('variables/3way_pre/data_set_0_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/3way_pre/data_set_4_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/3way_pre/data_set_5_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/3way_pre/data_set_6_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/3way_pre/data_set_7_gdc.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                else:
                    if FEATURES == 0:
                        with open('variables/mci_ad/data_set_0.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 1:
                        with open('variables/mci_ad/data_set_1.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 2:
                        with open('variables/mci_ad/data_set_2.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 3:
                        with open('variables/mci_ad/data_set_3.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 4:
                        with open('variables/3way_pre/data_set_4.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 5:
                        with open('variables/3way_pre/data_set_5.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 6:
                        with open('variables/3way_pre/data_set_6.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                    elif FEATURES == 7:
                        with open('variables/3way_pre/data_set_7.pickle', 'wb') as data_set:
                            pickle.dump([used_features, A, X, y, eigenvalues, eigenvectors, laplacians], data_set)
                            
    return used_features, A, X, y, eigenvalues, eigenvectors, laplacians