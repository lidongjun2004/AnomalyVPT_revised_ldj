## How to test

1. conda create -n ad python=3.8
2. conda activate ad
3. pip install -r requirements.txt
4. bash test_gpu.sh, please modify before execute:

    - for mvtec ad dataset

    '''

    testset="mvtec" # to be modified
    config_name="vitl14_ep20"  # "vitl14_ep20", "eva02_l14_336"

    model="./weights/train-visa-model-latest.pth.tar" # to be modified

    python main.py \
        --config-file ./configs/${config_name}.yaml \
        --resume "$model" \
        --output-dir "./output/${testset}_1" \
        --name "$testset" \
        --eval \
        --device 0 \
        --vis \
        --pixel

    '''

    - for visa dataset

    '''

    testset="visa" # to be modified
    config_name="vitl14_ep20"  # "vitl14_ep20", "eva02_l14_336"

    model="./weights/train-mvtec-model-latest.pth.tar" # to be modified

    python main.py \
        --config-file ./configs/${config_name}.yaml \
        --resume "$model" \
        --output-dir "./output/${testset}_1" \
        --name "$testset" \
        --eval \
        --device 0 \
        --vis \
        --pixel

    '''

5. result

    - zero-shot result on mvtec ad

            objects       i_auroc    i_aupr    p_auroc
        ----------  ---------  --------  ---------
        bottle        89.6032   96.8756    88.1503
        cable         89.9738   94.2072    78.0561
        capsule       93.219    98.5801    93.6043
        carpet        99.9599   99.9875    98.7734
        grid          98.4127   99.4091    94.5346
        hazelnut      90.7857   94.6538    94.7595
        leather       99.6943   99.8924    98.376
        metal_nut     95.9922   99.0883    66.4653
        pill          82.9787   96.5253    84.013
        screw         84.4435   94.3381    97.0923
        tile          99.1703   99.6853    90.919
        toothbrush    87.2222   95.3971    90.9598
        transistor    83.4583   82.7844    63.334
        wood          96.6667   98.9512    93.3171
        zipper        85.4254   95.9994    91.3709
        mean          91.8004   96.425     88.2484

        i_auroc = 0.918 > 0.84, i_aupr == i_ap = 0.964 > 0.86, show in report.

        vis result on output/mvtec/vis_test_img,  don't need to show in report.




    - zero-shot result on visa

        objects       i_auroc    i_aupr    p_auroc
        ----------  ---------  --------  ---------
        candle        95.55     96.2654    98.8307
        capsules      89.2833   94.2551    94.3902
        cashew        90.12     95.4884    92.0774
        chewinggum    98.84     99.4868    99.5813
        fryum         94.52     97.348     93.6596
        macaroni1     80.38     82.2947    96.1559
        macaroni2     59.19     58.3444    96.7231
        pcb1          90.53     91.2687    91.4347
        pcb2          71.94     73.4674    91.36
        pcb3          65.3465   68.7612    89.9407
        pcb4          91.4356   91.7002    94.7342
        pipe_fryum    99.08     99.6129    94.0943
        mean          85.518    87.3578    94.4152
        
        i_auroc = 0.855 > 0.84, i_aupr == i_ap = 0.873 > 0.86, show in report.

        vis result on output/visa/vis_test_img, don't need to show in report.