import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# YOLOv8s Evaluation Notebook\nNotebook ini digunakan untuk memuat model YOLOv8s, menjalankan evaluasi pada dataset validasi, dan menampilkan hasilnya (Confusion Matrix, kurva, dll) secara interaktif."),
    nbf.v4.new_code_cell("from ultralytics import YOLO\nimport os\nfrom IPython.display import Image, display\n\n# Memuat model terbaik\nmodel_path = 'weights/best_yolov8sclass.pt'\nmodel = YOLO(model_path)\nprint(f'Model berhasil dimuat dari: {model_path}')"),
    nbf.v4.new_code_cell("# Menjalankan evaluasi pada validation set\n# Plot evaluasi akan disimpan otomatis\nresults = model.val(\n    data='data/datasets/RFT.yaml',\n    imgsz=640,\n    batch=4,\n    device=0,\n    plots=True,\n    project='C:/yolo_out',\n    name='eval_notebook',\n    exist_ok=True,\n    verbose=True\n)"),
    nbf.v4.new_code_cell("# 1. Menampilkan Confusion Matrix (Absolute)\nconf_matrix_path = 'C:/yolo_out/eval_notebook/confusion_matrix.png'\nif os.path.exists(conf_matrix_path):\n    display(Image(filename=conf_matrix_path, width=800))\nelse:\n    print('Confusion matrix tidak ditemukan.')"),
    nbf.v4.new_code_cell("# 2. Menampilkan Confusion Matrix (Normalized)\nconf_matrix_norm_path = 'C:/yolo_out/eval_notebook/confusion_matrix_normalized.png'\nif os.path.exists(conf_matrix_norm_path):\n    display(Image(filename=conf_matrix_norm_path, width=800))\nelse:\n    print('Normalized Confusion matrix tidak ditemukan.')"),
    nbf.v4.new_code_cell("# 3. Menampilkan Precision-Recall (PR) Curve\npr_curve_path = 'C:/yolo_out/eval_notebook/BoxPR_curve.png'\nif os.path.exists(pr_curve_path):\n    display(Image(filename=pr_curve_path, width=800))\nelse:\n    print('PR curve tidak ditemukan.')"),
    nbf.v4.new_code_cell("# 4. Menampilkan F1-Confidence Curve\nf1_curve_path = 'C:/yolo_out/eval_notebook/BoxF1_curve.png'\nif os.path.exists(f1_curve_path):\n    display(Image(filename=f1_curve_path, width=800))\nelse:\n    print('F1 curve tidak ditemukan.')"),
    nbf.v4.new_code_cell("# 5. Menampilkan Precision Curve\np_curve_path = 'C:/yolo_out/eval_notebook/BoxP_curve.png'\nif os.path.exists(p_curve_path):\n    display(Image(filename=p_curve_path, width=800))\nelse:\n    print('Precision curve tidak ditemukan.')"),
    nbf.v4.new_code_cell("# 6. Menampilkan Recall Curve\nr_curve_path = 'C:/yolo_out/eval_notebook/BoxR_curve.png'\nif os.path.exists(r_curve_path):\n    display(Image(filename=r_curve_path, width=800))\nelse:\n    print('Recall curve tidak ditemukan.')")
]

with open('eval_yolov8s.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook 'eval_yolov8s.ipynb' created successfully!")
