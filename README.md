------------- A. Steps to run the system for operational forecast 

1. Copy all input data under each location of ./input. Note that the operational mode support
   only 2 types of runs: one is forecast mode that requires only an initial condition, 
   and the other is the detection mode that requires the entire global forecast (much like
   downscaling). 

2. Go to operation directory and run:
   - forecast mode: sh job_main.sh 2025112500 forecast gfs
   - detection mode: sh job_main.sh 2025112506 detection gfs

3. Check the output under ./output/postprocess/$cycle


------------- B. Steps to run the system for climate reconstruction

1. Run pre-process to create extended domains for each type of dataset era5/merra2/ifs/gfs/...
   This step will be needed for all modes including forecast, finetune, pre-train, or reconstruction.
   - need to edit config.json
   - add input data into a corresponding directory for each separate mode (forecast/finetune/pretrain)
   - output will be saved under ./output/ERA5_extend (or XXX_extend, depending on the input type)
   - user may change the output name for each corresponding experiment, e.g., ERA5_extend_8085. Need
     to change the corresponding part in the config.json with the new experiment name though.

2. Run inference/prediction step under models/prediction. Need to edit the following files
    - config.jon to choose the right inputs for input and output dir under SLICING_WINDOW var
    - ./models/prediction/map_slide.py --dataset era5
    - ./models/prediction/main.py (will call dynamic.py)

3. Run post process under the postprocess to display the data (should be under output/ERA5_slide) 
    - edit config.jon to make sure the script points to the right output from the prediction step
    - run plot_prediction.py/...

------------ C. Steps to run the system for pre-training and/or finetuning

1. Run pre-process to create extended domains for each type of dataset era5/merra2/ifs/gfs/...
   This step will be needed for all modes including forecast, finetune, pre-train, or reconstruction.
   - need to edit config.json 
   - add input data into a corresponding directory for each separate mode (forecast/finetune/pretrain)
   - output will be saved under ./output/ERA5_extend (or XXX_extend, depending on the input type)
   - user may change the output name for each corresponding experiment, e.g., ERA5_extend_8085. Need
     to change the corresponding part in the config.json with the new experiment name though.

2. Run extract domain to create positive/negative set of TC domain around a given ibtract data
   - run: python Extract_DynamicDomain.py --dataset era5
   - run: python Extract_PastDomain.py --dataset era5 (the output ERA5_positive produced by this step
     will override the ERA5_positive produced by the previous step Extract_DynamicDomain.py, as they 
     both produce the same positive output at t = 0)
   - soft link ERA5_dynamics -> DynamicsDomain,  ERA5_past -> PastDomain, and ERA5_positive -> POSITIVE 
     under the output. This soft link will be handled by the job_pretrain.sh or job_finetune.sh later.
   - NOTE: the whole step 2 (and steps 3-4 below ) will be skipped when running in the operational mode  

3. Run script to generate CSV file for train/test/val under either finetune or resnet18 
   - edit data paths and run data_organize.py (generate_csv.py): create file all.csv (should not have any POSITIVE file 
     under ./output/DyanmicDomain and ./output/PastDomain), which will be saved under output/csv/ 
   - edit data paths and run data_prepare.py (split_training_data.py): create train/test/val sets under output/csv/DynamicRemain_rusXX

4. To create a pre-trained model from scratch, 
    - edit the train_job.py for each lead time, with the correct input/output directories 
    - run job_sbatch.sh, whose output will be saved under 
      pre-trained/dynamic/ResNet_rus30_w15/Step_x/checkpoints/ 

   To fine-tune a pre-trained model, 
    - edit ./models/finetune/finetune_job.py, fixing all input/output directories 
    - run job_sbatch.sh

5. Run inference under post-process. Need to edit the following files
    - config.jon to choose the right inputs for input and output dir under SLICING_WINDOW var
    - ./models/prediction/map_slide.py --dataset era5
    - ./models/prediction/main.py (will call dynamic.py)

6. Run post process to display the data (should be under output/ERA5_slide) under postprocess plot_prediction.py

7. trainining/finetuning with different vars/levels can be done inside ./models/lib/Dataset/Merra2_dataset.py (should be on 
   config.json)


