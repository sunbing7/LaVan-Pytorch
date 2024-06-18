#python main.py --option=test --workers=4 --cuda --target=391 --iter=500 --data=imagenet --num_sample=100 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --netClassifier=resnet50


#python main.py --option=attack --workers=4 --cuda --target=512 --iter=500 --data=imagenet --num_sample=100 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --netClassifier=resnet50

#python main.py --option=attack --workers=4 --cuda --target=176 --iter=1 --data=imagenet --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --netClassifier=resnet50


#for TARGET_CLASS in {391,955,117,49,186,176,251,50}
#do
#    python main.py --option=attack --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --num_sample=1000 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --netClassifier=resnet50
#done

for TARGET_CLASS in {391,955,117,49,186,176,251,50}
do
    python main.py --option=adaptive_attack --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --model_name=resnet50_imagenet_finetuned_repaired.pth --num_sample=500 --x_min=176 --x-max=224 --y-min=176 --y_max=224 --epsilon=5 --plot --netClassifier=resnet50
done

#for TARGET_CLASS in {391,955,117,49,186,176,251,50}
#do
#    python main.py --option=test --workers=4 --cuda --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --netClassifier=resnet50
#done

#for TARGET_CLASS in {391,955,117,49,186,176,251,50}
#do
#    python main.py --option=test --workers=4 --cuda --model_name=resnet50_imagenet_finetuned_repaired.pth --target=$TARGET_CLASS --iter=1 --data=imagenet --x_min=176 --x-max=224 --y-min=176 --y_max=224 --plot --netClassifier=resnet50
#done