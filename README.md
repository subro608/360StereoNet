# Implementation Details
'main.py' -> 'main.py' that trains, validates and tests the model and also saves the validation-loss, optimizer-state and model weights as checkpoints for each epoch.

'dataloader/Newdepthloader.py' -> loads the left and right ERP image files along with their corresponding depth map files and groups them into train, validation and test set.

'dataloader/Depthloader.py' -> file uses the train, validation and test filenames from 'Newdepthloader.py' and loads the ERP images and corresponding depth maps for preprocessing.

'Test_img.py' -> uses the saved checkpoints to load the model weights and then predicts the depth map using the pre-trained model for a given pair of stereoscopic ERP images.

'depth_checkpoint_0.tar' -> determines the saved checkpoint after training the model for 1st epoch whcih means epoch 0. The model here is trained without using Image inpainting to remove invalid values from the depth map images. The last part '_0' varies from 0 to 5 meaning that the model is saved as checkpoints from epoch 0 to epoch 5. are saved from epoch 0 to epoch 5.


'depth_new_checkpoint_0.tar' -> determines the saved checkpoint after training the model for 1st epoch whcih means epoch 0. The model here is trained by using Navier-Stokes Image inpainting to remove invalid values from the depth map images. The last part '_0' varies from 0 to 5 meaning that the model is saved as checkpoints from epoch 0 to epoch 5.

Bellow is the list of all files and folders used for implementing the model.
![alt text](ls_filelist2.png "File list").
For using the code, first download the datasets and then extract them. Then arrange them in folders and put them along with the rest of the code as shown above. Note that only the left, right ERP images and their corresponding depth available in '.exr' format are required.


# Dataset
The link to the dataset is https://vcl3d.github.io/3D60/. In order to access the dataset there is a request form that needs to be filled. After that when we are given permission we can then downlaod the data from Zenodo which is hosting the data https://vcl3d.github.io/3D60/download.html. The dataset is split into three volumes Right, Up and Central (Left-down) viewpoints. For our work we consider Right and Central viewpoints.
# Citation 
As the code for the model is taken from the PSMNet model, then if the code is used for publication purposes please cite the PSMNet paper using the below.
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}




