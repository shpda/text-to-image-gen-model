# text-to-image-gen-model
Text-to-Image Conditional Generative Model

-------------- evaluate pretrained models:

python3 train_pixelcnnpp.py --train=0 --model_checkpoint="pretrained/cifar_conditional.pt" --n_epochs=90 --conditioning="one-hot"

python3 train_pixelcnnpp.py --train=0 --model_checkpoint="pretrained/bert_epoch_2.pt" --n_epochs=2 --conditioning="bert"

-------------- train model:

python3 train_pixelcnnpp.py --n_epochs=5 --conditioning="glove" --batch_size=28 --dataset="imagenet32"

 file                       epochs     embedding         BPD_actrual   BPD
 cifar_conditional.pt           90     one-hot           3.04          3.04
 cifar_unconditional.pt         89     unconditional                   3.04
 imagenet_conditional.pt         3     one-hot                         3.73
 imagenet_unconditional.pt       4     unconditional                   3.75
 bert_epoch_2.pt                 2     bert

