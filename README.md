# Pure-C-implementation-for-Deep-Learning-models
This is a pure C implementation for Deep Learning Models including: LeNet5, AlexNet, VGG16, ResNet18 without any external libraries requirements.

Dataset: MNIST, Cifar10

# Run
### enter the source code directory, e.g. Network_Model/Parallel_Thread/VGG16/VGG16_cifar10

```$ cd Network_Model/Parallel_Thread/VGG16/VGG16_cifar10 ```

### Run MakeFile

```$ make```

### Run the compiled file main

```$ ./main```

```bash
└─Network_Model
    ├─Parallel_Thread
    │  ├─AlexNet
    │  │  ├─AlexNet_cifar10
    │  │  │  └─data
    │  │  │      └─cifar
    │  │  │          └─cifar-10-batches-bin
    │  │  └─AlexNet_mnist
    │  ├─LeNet5
    │  │  ├─LeNet5_cifar10
    │  │  │  └─data
    │  │  │      └─cifar
    │  │  │          └─cifar-10-batches-bin
    │  │  └─LeNet5_mnist
    │  ├─ResNet18
    │  │  ├─ResNet18_cifar10
    │  │  │  └─data
    │  │  │      └─cifar
    │  │  │          └─cifar-10-batches-bin
    │  │  └─ResNet18_mnist
    │  └─VGG16
    │      ├─VGG16_cifar10
    │      │  └─data
    │      │      └─cifar
    │      │          └─cifar-10-batches-bin
    │      └─VGG16_mnist
    └─Single_Thread
        ├─AlexNet
        │  ├─Alexnet_cifar10
        │  │  └─data
        │  │      └─cifar
        │  │          └─cifar-10-batches-bin
        │  └─Alexnet_mnist
        ├─LeNet5
        │  ├─LeNet-5
        │  └─LeNet-5_cifar10
        │      └─data
        │          └─cifar
        │              └─cifar-10-batches-bin
        ├─ResNet18
        │  ├─ResNet18_cifar10
        │  │  └─data
        │  │      └─cifar
        │  │          └─cifar-10-batches-bin
        │  └─ResNet18_mnist
        ├─Transformer
        │  └─VisionTransformer_mnist
        └─VGG16
            ├─VGG16_single_cifar10
            │  └─data
            │      └─cifar
            │          └─cifar-10-batches-bin
            └─VGG16_single_mnist
        ```

