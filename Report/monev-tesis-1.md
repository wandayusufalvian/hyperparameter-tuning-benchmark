## Tambahan : 


[0] refactor code 
Progress tulisan yang sudah selesai : Bab 5 dan Bab 6.1 (Hasil Prapremrosesan Data)
[1] sudah pernah menjalankan semua eksperimen, kecuali yang menggunakan BOHB. Namun, ada kendala yaitu waktu training yang sangat lama ( bisa lebih dari 70 menit) untuk 1 kali training. Kemudian untuk census dataset, eksperimen gagal karena dimensi data terlalu besar sehingga 8 GB RAM tidak cukup untuk load data. 
[2] solusi dari poin [1] adalah melakukan data preprocessing ulang dengan mengelompokkan nilai-nilai kategori sehingga ketika dilakukan one hot encoding dimensi data yang dihasilkan tidak terlalu besar. Pengelompokan dilakukan selama tidak kehilangan konteks dari data fitur tersebut
[3] masih mempertimbangkan apakah akan tetap menjalankan eksperimen di local machine atau pindah ke google colab. kekurangan terbesar di local machine adalah ketika run pada iterasi-iterasi tengah dan akhir, waktu eksekusi semakin lama. hipotesis saya karena CPU
