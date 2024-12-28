# 25 0.09803 30 0.117647
CUDA_VISIBLE_DEVICES=3 python3 main.py --model Model_du --patch_size 128 --noise_std 0.09803 --n_color 1  --save retain_25_gray_du/25_gray_32_nq_du1 --init_channels 32 --layers 6 --nodes 5 --genotype nq_du1
CUDA_VISIBLE_DEVICES=2 python3 main2.py --model Model_du --patch_size 128 --noise_std 0.09803 --n_color 1  --save retain_25_gray_du/25_gray_32_nq_du2 --init_channels 32 --layers 6 --nodes 5 --genotype nq_du2
CUDA_VISIBLE_DEVICES=1 python3 main1.py --model Model_du --patch_size 128 --noise_std 0.09803 --n_color 1  --save retain_25_gray_du/25_gray_32_nq_du_fea1_1_channel --init_channels 32 --layers 6 --nodes 5 --genotype nq_du_fea1
CUDA_VISIBLE_DEVICES=0 python3 main3.py --model Model_du --patch_size 128 --noise_std 0.09803 --n_color 1  --save retain_25_gray_du/25_gray_32_nq_du_best --init_channels 32 --layers 6 --nodes 5 --genotype nq_du_best

