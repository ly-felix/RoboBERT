# RoboBERT
This is the official implementation of RoboBERT, which is a novel end-to-end multiple-modality robotic operations training framework. It allows an instruction-following agent, switching the policy by natural language input. The model itself is an extension or improvement of the traditional CNN-based diffusion policy, adding some creative training methods. For examples, two-stage training separates learning of different modalities; data-augmentation reduces the reliance of extra data, etc. The training framework has been tested on changllenging CALVIN benchmark and get the SOTA comparing to models without using large pretraining or extra dataset on ABCD -> D. Additionally, it also only use the trajectories with language labels. Based on these reasons, the model is much more easily to train. The repository shows the training/testing source code for repeative experiments. If you want to learn more about this work, please check our paper. Thank you!

## Model Structure

## Usage
Although the project and related libraries have been confirmed to run successfully on Windows, it is found that some libraries like Pyhash is difficult to compile and some performance loss for the model may also occurr on Windows, Linux is strongly recommended.

### Downloading Dataset and Pertraining
Please downloading the dataset on [CALVIN](https://github.com/mees/calvin) and BERT encoder on [BERT](https://huggingface.co/google-bert/bert-base-uncased/tree/main). 

### Cloning this Repository and Configurating Environment
```bash
git clone https://github.com/PeterWangsicheng/RoboBERT
cd RoboBERT
conda create --name RoboBERT python=3.8
conda activate RoboBERT
pip install -r requirements.txt
```
Then, modifing the pathes in config_path.json, replacing pathes of CALVIN and BERT as where they are in your computer.

### Extracting the Actions from Dataset
Because the training reads actions data, which is not very large in total, but more frequently than image data. It can reduce the I/O frequency and extend the harddisk life if all the action data can be restored in RAM before training cycle. We have created a script sparating or extracting the actions data individually from original CALVIN dataset and restoring into a pkl file called    ```dataset_wo_image_{dataset_name}.pkl ```.

```bash
python sparate_action_data.py --dataset_name ABCD_D --sparate_mode language
```
```--sparate_mode language``` means only sparate the trajectories with language labels. During the training, the RGB observations and actions are read from original CALVIN and extracted file respectively. 

You can also download the .pkl file we have prepared. Whatever which methods, you need modify the ```dataset_wo_image_path``` path in ```config_path.json``` to the path containing the corresponding pkl files. 

### Training the Model
After all the files mentioned are prepared and their pathes are modified in ```config_path.json```, we can train our model. Taking ABCD_D subset in CALVIN as an example, you need conduct the first-stage training by using standard language.

```bash
cd my_models/DDP_training
python training.py --world_size 2 --training_mode first --dataset_name ABCD_D
```

It will generate a checkpoint called ```model-9-first-ABCD_D.pt``` under the ```ckpt``` folder. Then we need to conduct the second stage training like

```bash
python training.py --world_size 2 --training_mode second --dataset_name ABCD_D --checkpoint ./model-9-first-ABCD_D.pt
```

The final mode is output as ```model-4-second-ABCD_D.pt``` in ```ckpt``` folder.
