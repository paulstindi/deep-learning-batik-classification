# Deep learning batik classification

Indonesian Batik classification using **VG16 convolutional neural network (CNN)** as feature extractor & softmax as classifier.

Dataset consist of 5 batik classes where each images will **belong to exactly one class**:

1. Parang: parang-like (some kind of traditional blade) motifs
1. Lereng: also blade-like pattern but less pointy than Parang
1. Ceplok: repetition of geometrical motif shapes (eg. rectangle, square, ovals, stars)
1. Kawung: kawung (fruit) motif
1. Nitik: flowers-like motifs

## Found issues

* Lereng and Parang motifs are alike (many redundant training data)

* There quite number of samples in current dataset which actually consist of multiple of motifs (mixed) (eg. Parang + Kawung, Ceplok + Kawung)

* Variances in datasets: variances in image's size & layout
