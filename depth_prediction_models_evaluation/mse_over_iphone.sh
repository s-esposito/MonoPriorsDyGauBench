# python mse_over_iphone.py --whole_image 0 --mean_scsh 0 --smoothing 1 --save_boxplot 1 --absrel 1
# python mse_over_iphone.py --whole_image 0 --mean_scsh 1 --smoothing 1 --save_boxplot 1 --absrel 1
# python mse_over_iphone.py --whole_image 1 --mean_scsh 0 --smoothing 1 --save_boxplot 1 --absrel 1
# python mse_over_iphone.py --whole_image 1 --mean_scsh 1 --smoothing 1 --save_boxplot 1 --absrel 1

python new_mse_over_iphone.py --whole_image 0 --mean_scsh 1 --smoothing 1 --save_boxplot 1 --absrel 1 --deltaone 1
python new_mse_over_iphone.py --whole_image 0 --mean_scsh 0 --smoothing 1 --save_boxplot 1 --absrel 1 --deltaone 1

python new_mse_over_iphone.py --whole_image 1 --mean_scsh 1 --smoothing 1 --save_boxplot 1 --absrel 1 --deltaone 1
python new_mse_over_iphone.py --whole_image 1 --mean_scsh 0 --smoothing 1 --save_boxplot 1 --absrel 1 --deltaone 1

python new_mse_over_iphone.py --whole_image 0 --mean_scsh 0 --smoothing 1 --save_boxplot 1 --absrel 1 --deltaone 1 --ransac 1
python new_mse_over_iphone.py --whole_image 1 --mean_scsh 0 --smoothing 1 --save_boxplot 1 --absrel 1 --deltaone 1 --ransac 1