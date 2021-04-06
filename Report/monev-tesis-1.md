## Yang sudah dilakukan : 


[0] Progress tulisan yang sudah selesai dan mohon untuk diperiksa : Bab V dan Bab 6.1 (Hasil Prapremrosesan Data)
[1] sudah menjalankan semua eksperimen, kecuali yang menggunakan BOHB. Namun, ada kendala yaitu waktu training yang sangat lama ( bisa lebih dari 70 menit) untuk 1 kali training. Kemudian untuk census dataset, eksperimen gagal karena dimensi data terlalu besar sehingga 8 GB RAM tidak cukup untuk load data. 
[2] solusi dari poin [1] adalah melakukan data preprocessing ulang dengan mengelompokkan nilai-nilai kategori sehingga ketika dilakukan one hot encoding dimensi data yang dihasilkan tidak terlalu besar. Pengelompokan dilakukan selama tidak kehilangan konteks dari data fitur tersebut.
[3] akan menjalankan seluruh eksperimen di google colab, bukan di local machine. Hal ini karena jika dilakukan di local machine ada kemungkinan terganggu dengan proses lain di lokal machine tersebut 
