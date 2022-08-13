- [HALTED_TASK] 
  - Completely use the pytorch's to do normalization. IT should happen in GPU. How I see that happening is, 
    when you read the tensors with torch, to transfer them to GPU at that point.
    Track time / performance between using pytorch's Augmentation and librosa's 

- [DONE] 
  - Generated Better Data.
  - Plot those sound spectrograms, And Rethink if you really do normalization thing. Seems like you don't 
  - Updated the Code to do an end to end training in Windows. 
  - NEED TO TAKE A LOOK AT HOW torch.Compose Actually Works


- [TODO] 
  [MODEL-TRAINING]
  - Add Configuration like from editable Yaml: 
    - train/test/overfit_check mode
    - model_save_path
    - file_path , meta_data path
    - sample_rate, batch_size


  - In the validation set, build the confusion matrix. 
  - Setup a easy training environment in colab. 

  [Model-Inference]
  - Run a inference on trained model. (This should be doable with training in parallel.
