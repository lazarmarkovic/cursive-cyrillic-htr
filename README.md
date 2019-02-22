# Cursive cyrillic handwritten text recognition project

Unpack following files into project folder: 
- https://drive.google.com/file/d/1J66qpSdfstA3Us5WpnxbcyAP3AekzAvb/view?usp=sharing


To run the main part of project use Jupiter NoteBook on files:
- Experiment SVM and KNN classification.ipynb
- Experiment CNN classification.ipynb

PS: Due to the lack of training data, no positive results are obtained :(

Problems unsolved:
- Good enough character segmentation (which is complex problem on its own accord)
- Maintenance of order of detected words when not written in one column
- Creation of BIG data set along with ligatures, almost no recognition is received either from SVM, KNN or CNN approaches 

References:
- https://github.com/Breta01/handwriting-ocr
- https://github.com/MonsieurV/py-findpeaks
- https://core.ac.uk/download/pdf/82703219.pdf
- https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
- https://bretahajek.com/2017/01/scanning-documents-photos-opencv/
- https://github.com/ftn-ai-lab/sc-2018-siit
