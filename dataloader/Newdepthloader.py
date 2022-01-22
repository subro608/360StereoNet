import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
#left_filepath = "/Volumes/Fast SSD/PSMNet-master/Matterport3D_left"
#right_filepath = "/Volumes/Fast SSD/PSMNet-master/Matterport3D_right"

def dataloader(left_filepath, right_filepath):
#sunCG_dataset
  new_left_img = []
  new_left_depth = []
  new_right_img = []

  file_list_left = os.listdir(left_filepath+"/SunCG_left")
  file_list_right = os.listdir(right_filepath+"/SunCG_right")

  area = ["area1/","area2/","area3/","area4/","area5a/","area5b/","area6/"]
  sun_left_depth = []
  sun_left_img = []
  for i in range(len(file_list_left)):
    if file_list_left[i].split("_")[5] == "0.0.exr" and file_list_left[i].split("_")[1] == "depth":
      sun_left_depth.append(file_list_left[i])
    elif file_list_left[i].split("_")[5] == "0.0.png":
      sun_left_img.append(file_list_left[i])
  sun_right_depth = []
  sun_right_img = []
  for i in range(len(file_list_right)):
    if file_list_right[i].split("_")[4] == "0.0.exr" and file_list_right[i].split("_")[1] == "depth":
      sun_right_depth.append(file_list_right[i])
    elif file_list_right[i].split("_")[4] == "0.0.png":
      sun_right_img.append(file_list_right[i])

  sun_img_left = []
  sun_img_right = []
  for i in range(len(sun_left_img)):
    for j in range(len(sun_right_img)):
      if sun_left_img[i].split("_")[0] == sun_right_img[j].split("_")[0]:
        sun_img_left.append(sun_left_img[i])
        sun_img_right.append(sun_right_img[j])
        break

  sun_img_left_depth = []
  for i in range(len(sun_img_left)):
    for j in range(len(sun_left_depth)):
      if sun_img_left[i].split("_")[0] == sun_left_depth[j].split("_")[0]:
        sun_img_left_depth.append(sun_left_depth[j])
        break

  for i in range(len(sun_img_left)):
    new_left_img.append(left_filepath+"/SunCG_left/"+sun_img_left[i])

  for i in range(len(sun_img_left_depth)):
    new_left_depth.append(left_filepath+"/SunCG_left/"+sun_img_left_depth[i])

  for i in range(len(sun_img_right)):
    new_right_img.append(right_filepath+"/SunCG_right/"+sun_img_right[i])

  print("suncg",len(sun_img_left),len(sun_img_left_depth))


  # Stanford Dataset
  stan_left_depth = []
  stan_left_img = []
  stan_right_depth = []
  stan_right_img = []


  for a in area:
    file_list_left = os.listdir(left_filepath+"/Stanford2D3D_left/"+a)
    file_list_right = os.listdir(right_filepath+"/Stanford2D3D_right/"+a)
    stan_left = []
    stan_right = []
    stan_depth = []
    for i in range(len(file_list_left)):
      if file_list_left[i].split("_")[7] == "0.0.exr" and file_list_left[i].split("_")[3] == "depth":
        stan_left_depth.append(file_list_left[i])
        stan_depth.append(file_list_left[i])
      elif file_list_left[i].split("_")[7] == "0.0.png":
        stan_left_img.append(file_list_left[i])
        stan_left.append(file_list_left[i])

    for i in range(len(file_list_right)):
      if file_list_right[i].split("_")[6] == "0.0.exr" and file_list_right[i].split("_")[3] == "depth":
        stan_right_depth.append(file_list_right[i])
      elif file_list_right[i].split("_")[6] == "0.0.png":
        stan_right_img.append(file_list_right[i])
        stan_right.append(file_list_right[i])

    stan_img_left = []
    stan_img_right = []
    for i in range(len(stan_left)):
      for j in range(len(stan_right)):
        if stan_left[i].split("_")[0] == stan_right[j].split("_")[0] and stan_left[i].split("_")[2] == stan_right[i].split("_")[2]:
          stan_img_left.append(stan_left[i])
          stan_img_right.append(stan_right[j])
          break

    stan_img_left_depth = []
    for i in range(len(stan_img_left)):
      for j in range(len(stan_depth)):
        if stan_img_left[i].split("_")[0] == stan_depth[j].split("_")[0] and stan_img_left[i].split("_")[2] == stan_depth[i].split("_")[2]:
          stan_img_left_depth.append(stan_depth[j])
          break


    for i in range(len(stan_img_left)):
      new_left_img.append(left_filepath+"/Stanford2D3D_left/"+a+stan_img_left[i])


    for i in range(len(stan_img_left_depth)):
      new_left_depth.append(left_filepath+"/Stanford2D3D_left/"+a+stan_img_left_depth[i])


    for i in range(len(stan_img_right)):
      new_right_img.append(right_filepath+"/Stanford2D3D_right/"+a+stan_img_right[i])

    print("stanford",len(stan_img_left),len(stan_img_left_depth))

  #Matterport3D Dataset
  file_list_left = os.listdir(left_filepath+"/Matterport3D_left")
  file_list_right = os.listdir(right_filepath+"/Matterport3D_right")


  left_depth = []
  left_img = []
  for i in range(len(file_list_left)):
    if file_list_left[i].split("_")[6] == "0.0.exr" and file_list_left[i].split("_")[2] == "depth":
      left_depth.append(file_list_left[i])
    elif file_list_left[i].split("_")[6] == "0.0.png":
      left_img.append(file_list_left[i])
  right_depth = []
  right_img = []
  for i in range(len(file_list_right)):
    if file_list_right[i].split("_")[5] == "0.0.exr" and file_list_right[i].split("_")[2] == "depth":
      right_depth.append(file_list_right[i])
    elif file_list_right[i].split("_")[5] == "0.0.png":
      right_img.append(file_list_right[i])

  img_left = []
  img_right = []
  for i in range(len(left_img)):
    for j in range(len(right_img)):
      if left_img[i].split("_")[1] == right_img[j].split("_")[1]:
        img_left.append(left_img[i])
        img_right.append(right_img[j])
        break

  img_left_depth = []
  for i in range(len(img_left)):
    for j in range(len(left_depth)):
      if img_left[i].split("_")[1] == left_depth[j].split("_")[1]:
        img_left_depth.append(left_depth[j])
        break

  for i in range(len(img_left)):
    new_left_img.append(left_filepath+"/Matterport3D_left/"+img_left[i])

  for i in range(len(img_left_depth)):
    new_left_depth.append(left_filepath+"/Matterport3D_left/"+img_left_depth[i])

  for i in range(len(img_right)):
    new_right_img.append(right_filepath+"/Matterport3D_right/"+img_right[i])

  print("Matterport3D",len(img_left),len(img_left_depth))  
  train_left = []
  train_right = []
  train_depth = []
  test_left = []
  test_right = []
  test_depth = []
  val_left = []
  val_right = []
  val_depth = []
  #Fold 1
  test_left.extend(new_left_img[:4986])
  test_right.extend(new_right_img[:4986])
  test_depth.extend(new_left_depth[:4986])
  #Fold 2
  train_left.extend(new_left_img[4986:9972])
  train_right.extend(new_right_img[4986:9972])
  train_depth.extend(new_left_depth[4986:9972])
  #Fold 3
  train_left.extend(new_left_img[9972:14958])
  train_right.extend(new_right_img[9972:14958])
  train_depth.extend(new_left_depth[9972:14958])
  #Fold 4
  train_left.extend(new_left_img[14958:19944])
  train_right.extend(new_right_img[14958:19944])
  train_depth.extend(new_left_depth[14958:19944])
  #Fold 5
  val_left.extend(new_left_img[19944:24930])
  val_right.extend(new_right_img[19944:24930])
  val_depth.extend(new_left_depth[19944:24930])
  print(len(train_left),len(val_left),len(test_left))
  return train_left, train_right, train_depth, test_left, test_right, test_depth, val_left, val_right,val_depth
