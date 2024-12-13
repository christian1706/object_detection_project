# path to your own data and coco file
train_data_dir = r"C:\Users\chris\Desktop\formations\pytorch\mini_coco_v2\train_images"
train_coco = r"C:\Users\chris\Desktop\formations\pytorch\mini_coco_v2\train_annotations.json"

val_data_dir = r"C:\Users\chris\Desktop\formations\pytorch\mini_coco_v2\val_images"
val_coco = r"C:\Users\chris\Desktop\formations\pytorch\mini_coco_v2\val_annotations.json"

saved_model = r"C:\Users\chris\Desktop\formations\pytorch\DL\frcnn_medium_sample\output-11-12-2024-18-41-26\epoch_2_model.pth"

test_image =  r"C:\Users\chris\Desktop\formations\pytorch\DL\test_image7.png"

# Batch size
train_batch_size = 1
val_batch_size = 1

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Two classes; Only target class or background
num_classes = 6
num_epochs = 5

lr = 0.00001
momentum = 0.9
weight_decay = 0.005
LR_SCHED_STEP_SIZE = 0.1
LR_SCHED_GAMMA = 0.1

classes = ["__background__","person", "dog", "cat", "house", "laptop"]