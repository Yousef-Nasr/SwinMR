'''
# -----------------------------------------
Select Dataset
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # ------------------------------------------------
    # CC-359 Calgary Campinas Public Brain MR Dataset
    # ------------------------------------------------
    # CC-SAG-NPI d.1.1
    if dataset_type in ['ccsagnpi']:
        from data.dataset_CCsagnpi import DatasetCCsagnpi as D
    # CC-SAG-PI d.1.1
    elif dataset_type in ['ccsagpi']:
        from data.dataset_CCsagpi import DatasetCCsagpi as D
    
    # ------------------------------------------------
    # Enhanced Dataset with DICOM/JPG support and comprehensive noise
    # ------------------------------------------------
    elif dataset_type in ['enhanced', 'enhanced_npi', 'dicom_jpg']:
        from data.dataset_enhanced import DatasetEnhanced as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
