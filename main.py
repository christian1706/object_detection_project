import os
import torch
from tqdm import tqdm
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import config
from utils import (
    get_model,
    collate_fn,
    get_transform,
    plot_loss,
    plot_image,
    inference,
    myOwnDataset
)

def setup_output_directory():
    current_dir = os.getcwd()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    output_dir_name = "output-" + dt_string
    output_dir = os.path.join(current_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_checkpoint(output_dir, model, optimizer):
    checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    if checkpoint_files:
        checkpoint_files.sort()
        latest_checkpoint = os.path.join(output_dir, checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_dict = checkpoint['loss_dict']
    else:
        start_epoch = 0
        loss_dict = {'train_loss': [], 'valid_loss': []}
    return start_epoch, loss_dict

def train_one_epoch(model, optimizer, train_data_loader, device):
    train_loss_list = []
    tqdm_bar = tqdm(train_data_loader, total=len(train_data_loader))
    for idx, data in enumerate(tqdm_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        train_loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

    return train_loss_list

def evaluate_one_epoch(model, val_data_loader, device):
    val_loss_list = []
    tqdm_bar = tqdm(val_data_loader, total=len(val_data_loader))
    for i, data in enumerate(tqdm_bar):
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            losses = model(images, targets)

        loss = sum(loss for loss in losses.values())
        val_loss_list.append(loss.item())
        tqdm_bar.set_description(desc=f"Validation Loss: {loss:.4f}")

    return val_loss_list

def initialize_model_and_optimizer(device):
    model = get_model(config.num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    return model, optimizer


def initialize_dataloaders():
    train_dataset = myOwnDataset(
        root=config.train_data_dir, annotation=config.train_coco, transforms=get_transform()
    )
    val_dataset = myOwnDataset(
        root=config.val_data_dir, annotation=config.val_coco, transforms=get_transform()
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )
    return train_data_loader, val_data_loader

def save_checkpoint(epoch, model, optimizer, loss_dict, output_dir):
    ckpt_file_name = os.path.join(output_dir, f"epoch_{epoch}_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_dict': loss_dict
    }, ckpt_file_name)


def plot_loss(train_loss, valid_loss, output_dir):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()

    train_ax.plot(train_loss, color='blue')
    train_ax.set_xlabel('Iteration')
    train_ax.set_ylabel('Training Loss')

    valid_ax.plot(valid_loss, color='red')
    valid_ax.set_xlabel('Iteration')
    valid_ax.set_ylabel('Validation Loss')

    figure_1.savefig(f"{output_dir}/train_loss.png")
    figure_2.savefig(f"{output_dir}/valid_loss.png")

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = setup_output_directory()
    train_data_loader, val_data_loader = initialize_dataloaders()
    model, optimizer = initialize_model_and_optimizer(device)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_SCHED_STEP_SIZE, gamma=config.LR_SCHED_GAMMA
    )

    start_epoch, loss_dict = load_checkpoint(output_dir, model, optimizer)

    for epoch in range(start_epoch, config.num_epochs):
        print(f"----------Epoch {epoch + 1}----------")

        train_loss_list = train_one_epoch(model, optimizer, train_data_loader, device)
        loss_dict['train_loss'].extend(train_loss_list)

        lr_scheduler.step()

        valid_loss_list = evaluate_one_epoch(model, val_data_loader, device)
        loss_dict['valid_loss'].extend(valid_loss_list)

        save_checkpoint(epoch + 1, model, optimizer, loss_dict, output_dir)
        plot_loss(loss_dict['train_loss'], loss_dict['valid_loss'], output_dir)

    with open(os.path.join(output_dir, "loss_dict.pkl"), "wb") as file:
        pickle.dump(loss_dict, file)

    print("Training Finished!")

if __name__ == "__main__":
    main()


