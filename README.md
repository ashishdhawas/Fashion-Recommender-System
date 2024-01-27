## Fashion Recommender System

***Project Overview:***


Developed a Streamlit Application which is a Fashion Recommender System. Users can upload an image of a fashion item, and the application extracts features using a pre-trained ResNet50 model. It then recommends similar fashion items based on the extracted features. The user interface includes a file upload option, the display of the uploaded image, and a set of recommended items with their respective images. The application provides error handling during file upload and could benefit from additional styling and user instructions for an enhanced experience. Overall, it's a functional and user-friendly fashion recommendation tool.



DataSet Link: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

***About DataSet:***

Each product is identified by an ID like 42431. You will find a map to all the products in styles.csv. From here, you can fetch the image for this product from images/42431.jpg and the complete metadata from styles/42431.json.

To get started easily, we also have exposed some of the key product categories and it's display name in styles.csv.

***Project Workflow:***

1. Import model: Import CNN Pre-trained ResNet50 Model.

2. Extract Features: Extract Features from a given dataset using ResNet50 model and store it in pickle file for further use.

3. Generate Recommendation: Generate recommendation using K-Nearest Neighbor(KNN) Algorithm.


***Requirements:***


To run the provided Streamlit app for the Fashion Recommender System, you'll need a set of dependencies installed in your Python environment. Here's a list of requirements:

**Python:** Ensure you have Python installed on your system. You can download it from python.org.

**Packages and Libraries:**

**streamlit:** The main library for creating the interactive web app.

**link:** https://docs.streamlit.io/get-started

**PIL:** Python Imaging Library, used for image processing.

**link:** https://pillow.readthedocs.io/en/stable/

**numpy:** For numerical operations.

**link:** https://numpy.org/doc/

**tensorflow:** The deep learning framework used for the ResNet50 model.

**How to setup TensorFlow 2.3.1 — CPU/GPU (Windows 10):**
**link:** https://arsanatl.medium.com/how-to-setup-tensorflow-2-3-1-cpu-gpu-windows-10-e000e7811e2b

**scikit-learn:** Used for Nearest Neighbors algorithm.

**link:** https://scikit-learn.org/stable/modules/neighbors.html

**os:** Standard Python library for interacting with the operating system.


**You can install these dependencies using the following command:**

pip install streamlit 


pip install Pillow 


pip install numpy 


pip install tensorflow 


pip install scikit-learn



**Pre-trained ResNet50 Model:** The app uses the ResNet50 model with pre-trained ImageNet weights. These weights are automatically downloaded by TensorFlow when you instantiate the model.

**keras pre-trained model link:** https://keras.io/api/applications/

**Image Dataset:** If you want to train and fine-tune the model or use a different model, you might need a labeled image dataset.

**link:** https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

**Miscellaneous:**

**A web browser:** Streamlit apps are accessed through a web browser.

**An internet connection:** Needed for downloading the ResNet50 weights during the first run.

**Recommended:**

**pickle:** Used for loading the pre-computed features and filenames from pickle files.

**tqdm:** For displaying progress bars during feature extraction (optional but useful for large datasets).

**you can install these dependencies using:**

pip install pickle5 


pip install tqdm



**Note:** The exact versions of these libraries may change over time, so it's a good practice to check for the latest versions and update accordingly. You can create a requirements.txt file with the required packages and versions for better project reproducibility.
