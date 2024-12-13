## Object Detetcion

# Group member names
    -  TAMDJO WAGFIN CHRISTIAN
    -  DONFACK NGUIMFACK STEVE


# 1. Training the model
      -- To train the model clone the followin git repository
      -- Prepare a datset in the coco format, the data set should be organized in the following way 
             -- train_images
               -- train_annotations.json
             -- validation_images
                -- validation_annotations.json
      -- Input the path of your data set into the config.py script at the respective positions
             -- train_data_dir
                -- train_coco
             -- val_data_dir
                -- val_coco
### NOTE:
     All modifications concerning the learning rate, the number of epochs, the number of epochs,
     the train and validation batch size are found in the config script

# 2. Testing the model
    -- To test our model use insert the path to your saved model in the save_model_dir variable in the config.py script
#### Hint: The saved model are found in an `/current_working_dir/output-d-m-y` folder
    -- Launch the test.py script 