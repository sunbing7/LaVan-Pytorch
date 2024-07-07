#############################################################################################################################################################################################
#resnet50 imagenet

#for TARGET_CLASS in {391,955,117,49,186,176,251,50}
#do
#    python main.py --option=attack --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --arch=resnet50
#done

#python main.py --option=attack --workers=4 --cuda --target=50 --iter=1 --data=imagenet --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --arch=resnet50

#for TARGET_CLASS in {391,955,117,49,186,176,251,50}
#do
#    python main.py --option=adaptive_attack --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --model_name=resnet50_imagenet_finetuned_repaired.pth --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --arch=resnet50
#done

#python main.py --option=adaptive_attack --workers=4 --cuda --target=391 --iter=1 --data=imagenet --model_name=resnet50_imagenet_finetuned_repaired.pth --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --arch=resnet50

#for TARGET_CLASS in {391,955,117,49,186,176,251,50}
#do
#    python main.py --option=test --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=resnet50
#done
#python main.py --option=test --workers=4 --cuda --target=391 --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=resnet50
#for TARGET_CLASS in {391,955,117,49,186,176,251,50}
#do
#    python main.py --option=test --workers=4 --cuda --model_name=resnet50_imagenet_finetuned_repaired.pth --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=resnet50
#done

#python main.py --option=test --workers=4 --cuda --model_name=resnet50_imagenet_finetuned_repaired.pth --target=391 --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=resnet50


#############################################################################################################################################################################################
#vgg19
#for TARGET_CLASS in {115,117,119,450,395,329,765,723}
#do
    #python main.py --option=attack --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=1 --plot --arch=vgg19
    #python main.py --option=test --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=vgg19
    #python main.py --option=test --workers=4 --cuda --model_name=vgg19_imagenet_finetuned_repaired.pth --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=vgg19
#done

#python main.py --option=attack --workers=4 --cuda --target=115 --iter=1 --data=imagenet --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=1 --plot --arch=vgg19

#############################################################################################################################################################################################
#googlenet 836,174,940,882,98,812,258,935
#for TARGET_CLASS in {836,174,940,882,98,812,258,935}
#do
    #python main.py --option=attack --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=1 --plot --arch=googlenet
    #python main.py --option=test --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=googlenet
    #python main.py --option=test --workers=4 --cuda --model_name=googlenet_imagenet_finetuned_repaired.pth --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=googlenet
#done

#python main.py --option=attack --workers=4 --cuda --target=836 --iter=1 --data=imagenet --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=3 --plot --arch=googlenet

#############################################################################################################################################################################################
#asl mobilenet 8,15,6,28,18,16,5,21
#for TARGET_CLASS in {8,15,6,28,18,16,5,21}
#do
    #python main.py --option=attack --workers=4 --cuda --model_name=mobilenet_asl.pth --target=$TARGET_CLASS --iter=1 --data=asl --num_sample=500 --x_min=100 --x-max=142 --y-min=100 --y_max=142 --epsilon=1 --plot --arch=mobilenet
    #python main.py --option=test --workers=4 --cuda --model_name=mobilenet_asl.pth --target=$TARGET_CLASS --iter=1 --data=asl --x_min=100 --x-max=142 --y-min=100 --y_max=142 --plot --arch=mobilenet
    #python main.py --option=test --workers=4 --cuda --model_name=mobilenet_asl_ae_repaired.pth --target=$TARGET_CLASS --iter=1 --data=asl --x_min=100 --x-max=142 --y-min=100 --y_max=142 --plot --arch=mobilenet
#done

#python main.py --option=attack --workers=4 --cuda --model_name=mobilenet_asl.pth --target=28 --iter=1 --data=asl --num_sample=500 --x_min=100 --x-max=142 --y-min=100 --y_max=142 --epsilon=5 --plot --arch=mobilenet
#python main.py --option=test --workers=4 --cuda --model_name==mobilenet_asl_ae_repaired.pth --target=8 --iter=1 --data=asl --x_min=100 --x-max=142 --y-min=100 --y_max=142 --plot --arch=mobilenet
#############################################################################################################################################################################################
#caltech shufflenetv2
#for TARGET_CLASS in {7,8,89,36,35,92,78,9}
#do
    #python main.py --option=attack --workers=4 --cuda --model_name=shufflenetv2_caltech.pth --target=$TARGET_CLASS --iter=1 --data=caltech --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --arch=shufflenetv2
    #python main.py --option=test --workers=4 --cuda --model_name=shufflenetv2_caltech.pth --target=$TARGET_CLASS --iter=1 --data=caltech --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=shufflenetv2
    #python main.py --option=test --workers=4 --cuda --model_name=shufflenetv2_caltech_finetuned_repaired.pth --target=$TARGET_CLASS --iter=1 --data=caltech --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --arch=shufflenetv2
#done

#python main.py --option=attack --workers=4 --cuda --model_name=shufflenetv2_caltech.pth --target=9 --iter=1 --data=caltech --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --arch=shufflenetv2

#############################################################################################################################################################################################
#eurosat resnet50
for TARGET_CLASS in {2,8,6,4,7,9,3,0}
do
    #python main.py --option=attack --workers=4 --cuda --model_name=resnet50_eurosat.pth --target=$TARGET_CLASS --iter=1 --data=eurosat --num_sample=500 --x_min=50 --x-max=64 --y-min=50 --y_max=64 --epsilon=5 --plot --arch=resnet50
    python main.py --option=test --workers=4 --cuda --model_name=resnet50_eurosat.pth --target=$TARGET_CLASS --iter=1 --data=eurosat --x_min=50 --x-max=64 --y-min=50 --y_max=64 --plot --arch=resnet50
    python main.py --option=test --workers=4 --cuda --model_name=resnet50_eurosat_finetuned_repaired.pth --target=$TARGET_CLASS --iter=1 --data=eurosat --x_min=50 --x-max=64 --y-min=50 --y_max=64 --plot --arch=resnet50
done

#python main.py --option=attack --workers=4 --cuda --model_name=resnet50_eurosat.pth --target=0 --iter=1 --data=eurosat --num_sample=500 --x_min=50 --x-max=64 --y-min=50 --y_max=64 --epsilon=1 --plot --arch=resnet50

#############################################################################################################################################################################################
