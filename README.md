# **Gear Classification Model 🔧📸**
*A Deep Learning-based Image Classification Model for Gear Maintenance Analysis*

![GitHub](https://img.shields.io/badge/Status-Active-brightgreen.svg)  
![GitHub](https://img.shields.io/badge/Framework-PyTorch-red.svg)  
![GitHub](https://img.shields.io/badge/Deployment-Hugging%20Face-orange.svg)  
![GitHub](https://img.shields.io/badge/ML%20Model-ResNet18-blue.svg)  

---

## **📌 Problem Statement**
### **Optimizing Gear Quality Control in Manufacturing Using AI-Powered Image Classification**
In modern manufacturing, ensuring the quality of gears is critical for **operational efficiency** and **product reliability**. However, manual inspection processes are prone to **human error**, **subjectivity**, and **inefficiencies**, leading to increased costs and potential failures in downstream applications.

### **🛠 Solution**
This project aims to develop an **AI-powered image classification model** that automates the identification of **damaged, rusted, and undamaged gears** using deep learning techniques. By leveraging a robust dataset and deploying a web-based application, the solution enables manufacturers to:
- **Quickly assess gear conditions**
- **Minimize downtime**
- **Improve overall production quality**

### **🔍 Key Challenges Addressed**
✔ **Inconsistent Quality Control** → Eliminates subjective manual inspections and ensures uniform quality assessment.  
✔ **Reduced Efficiency** → Automates defect detection to accelerate production processes.  
✔ **Minimized Costs** → Identifies defective gears early, reducing maintenance expenses and rework.  

Our **deep learning model**, trained on real-world gear images, classifies gear conditions with high accuracy, ensuring manufacturers can **streamline quality assurance processes** and **reduce operational risks**.

---

## **🚀 Features**
✔ **Image Classification**: Classifies images into three categories – *Rusty Gears, Damaged Gears, and Undamaged Gears*.  
✔ **Pretrained ResNet-18 Model**: Utilizes transfer learning for improved accuracy.  
✔ **Web App**: Built using **Gradio**, allowing users to upload images for real-time classification.  
✔ **Model Deployment**: Hosted on **Hugging Face Spaces** for public access.  
✔ **Automated Recommendation System**: Suggests *Maintenance Required* for rusted gears and *Replacement Needed* for damaged gears.  

---

## **🔧 Tech Stack**
- **Programming Language**: Python  
- **Deep Learning Framework**: PyTorch  
- **Model Architecture**: ResNet-18 & CNN  
- **Deployment**: Hugging Face Spaces  
- **Web Interface**: Gradio  
- **Other Libraries**: OpenCV, NumPy, Pandas, Matplotlib  

---

## **📂 Project Structure**
📦 Gear-Classification-Model 
│── app.py # Main script for web application 

│── model_training.ipynb # Jupyter Notebook for training the model

│── requirements.txt # Dependencies for running the project 

│── README.md # Documentation (this file) 

│── data/ # Dataset

│── models/ # Contains trained models 

│── assets/ # Sample images for testing

echo "## 📝 Model Performance

- **Training Accuracy**: **92%**
- **Testing Accuracy**: **100%** _(Experimental fine-tuning beyond 92% was done for learning purposes.)_

---

## 📜 License

This project is licensed under the **MIT License** – feel free to modify and distribute.

---

## 📞 Contact

For any queries or contributions, feel free to connect:

📧 **Email**: rushikesh.pokale@edhec.com  
🔗 **Portfolio**: [RushikeshPokale](https://www.datascienceportfol.io/rushikeshpokale)  
🤝 **LinkedIn**: [Rushikesh Pokale](https://www.linkedin.com/in/rushikesh-pokale/)

🚀 **If you find this project useful, give it a ⭐ on GitHub!**  

---

## **📥 Model Download**
The trained model is hosted on **Hugging Face**. To use it, download the model using:
```python
import gdown
gdown.download("https://huggingface.co/rushikesh830/gear-classification-model/blob/main/gear_classifier.pkl", "gear_classifier.pkl", quiet=False)







