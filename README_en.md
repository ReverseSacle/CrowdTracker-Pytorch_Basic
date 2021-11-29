**FairMOT_Pytorch_Tracker(Single Camera)**
===
[简体中文](https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic/blob/main/README.md) | [English](https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic/blob/main/README_en.md)

Address Navigation
---
+ [->Paddle_edtion_address](https://github.com/ReverseSacle/FairMOT-Paddle-Tracker_Basic)
+ [->Original_author_Github_address](https://github.com/ifzhang/FairMOT)

Preview
---
![MOT20-01](https://github.com/ReverseSacle/FairMOT_Paddle/blob/main/docs/MOT20-01.gif)

Preview for Interface
---
![Interface](https://user-images.githubusercontent.com/73418195/126273708-42a9aec3-a07f-4102-aaf2-3a6f5cadf2b5.png)



Enviroment Requirement
---
+ Python3
+ OpenCV-python
+ DCNv2
+ Needed requirements -> All the requirements  in [->Original_author_Github_address](https://github.com/ifzhang/FairMOT)
+ Test system -> window10
+ The provided pkged enviroment(coda enviroment wich have all the needed libs) --> [->Google Drive](https://drive.google.com/file/d/1xNADf_ARQnDhKNx1rEOHgXszG2lrSEet/view?usp=sharing)

Introduction
---
+ [->Making_Introduction](https://github.com/ReverseSacle/FairMOT_paddle/blob/main/docs/Making_Introduction_en.md)

Provide Model file
---
+ **Download：** Provided by Original author[->Google Drive](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view) -> need to put them in  the folder named __models__(root_dir)
+ **extra Missing file** [->Google Drive](https://drive.google.com/file/d/1sZ0PHOtHkfAHpJ1Na4Ff0SD7NJktFKHq/view?usp=sharing) -> put it in ```C:\Users\User name\.cache\torch\hub\checkpoints```


Quickly start
---
+ ```git clone "https://github.com/ReverseSacle/FairMOT-Pytorch-Tracker_Basic.git"```
+ Unzip Fairmot_env in Anaconda3/envs/ folder
+ Use pycharm,choose Fairmot_env enviroment.Then,create a folder named **models**,unzip the file in the __models__ folder.

About Function of Buttons
---
+ [->What the mean of the button](https://github.com/ReverseSacle/FairMOT-Paddle-Tracker_Basic/blob/main/docs/The_button_function_en.md)

Update Record
---
2021.11.29  Create a new brach ByteTrack-kernel，replace original mot kernel with bytetrack.
