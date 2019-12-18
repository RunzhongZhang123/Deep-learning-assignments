README file for 
   assignment0
for Course ECBM E4040.
Updated 8/26/2019

The core of the assignment is described in jupyter notebook file called
   Assignment0.ipynb

Organization of the directory



├── README.txt
├── Assignment 0.ipynb          # Main jupyter notebook for assignment0
├── requirements.txt            # This file contains all the required dependecies for the Homework. Always keep 
│                               #a requirements.txt file for your projects. To generate use`pip freeze > requirements.txt`
├── pics                        # Pictures
│     ├── deadpool.jpg          #   
│     ├── scope_error.png       #
│     └── university.jpg        #
├── tmp                         #
│    └── data                   # Contains data. Here the tmp folder is created after you run the code to download the data
│         ├── t10k-images-idx3-ubyte.gz #
│         ├── t10k-labels-idx1-ubyte.gz #
│         ├── train-images-idx3-ubyte.gz#
│         └── train-labels-idx1-ubyte.gz#
└── utils                       # Directory with utility functions
     ├── classifiers            # Directory with classifier functions
     │    ├── basic_classifiers.py  #
     │    ├── linear_svm.py     #
     │    └── softmax.py        #
     └── cifar_utils.py         #     