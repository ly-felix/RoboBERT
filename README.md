# RoboBERT
This is the official implementation of RoboBERT, which is a novel end-to-end multiple-modality robotic operations training framework. It allows an instruction-following agent, switching the policy by natural language input. The model itself is an extension or improvement of the traditional CNN-based diffusion policy, adding some creative training methods. For examples, two-stage training separates learning of different modalities; data-augmentation reduces the reliance of extra data, etc. The training framework has been tested on changllenging CALVIN benchmark and get the SOTA comparing to models without using large pretraining or extra dataset on ABCD -> D. Additionally, it also only use the trajectories with language labels. Based on these reasons, the model is much more easily to train. The repository shows the training/testing source code for repeative experiments. If you want to learn more about this work, please check our paper. Thank you!

## Model Structure

## Usage
Although the project and related libraries have been confirmed to run successfully on Windows, it is found that some libraries like Pyhash is difficult to compile and some performance loss for the model may also occurr on Windows, Linux is strongly recommended.

### Downloading Dataset and Configurating Environment
Please downloading the dataset on [CALVIN](https://github.com/mees/calvin) and BERT encoder on [BERT](https://huggingface.co/google-bert/bert-base-uncased/tree/main). Please replace the corresponding pathes in config_path.json. Note that the CALVIN path should be the upper-level directory of the "task_ABCD_D" or other datasets. You should also install all the components under three CALVIN diectories by pip install -e .

### Extracting the Actions from Dataset
Because the training reads actions data, which is not very large in total, more frequently than image data, it can reduce the I/O frequency and extend the harddisk life if all the action data can be restored in RAM before training cycle. We have created a script extracting the actions data individually from original CALVIN dataset and restoring into a pkl file called dataset_wo_image_{dataset_name}.pkl. 

```bash
python sparate_action_data.py --dataset_name ABCD_D --sparate_mode language
```
After the extraction completing, please modify dataset_wo_image_path in config_path.json to the path of dataset_wo_image_{dataset_name}.pkl.

During the training, the RGB observations and actions are read from original CALVIN and extracted file respectively. 

### Training the Model
After completing the two steps above, 

### Evaluating the Model


