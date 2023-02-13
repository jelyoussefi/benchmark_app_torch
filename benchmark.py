
import argparse
import numpy as np
import time
import torch
import torchvision
from torchvision import models

import torch.backends.cudnn as cudnn
#cudnn.benchmark = True

from torch_ort import ORTInferenceModule, OpenVINOProviderOptions 


def rn50_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

# decode the results into ([predicted class, description], probability)
def predict(img_path, model):
    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    #if torch.cuda.is_available():
    #    input_batch = input_batch.to('cuda')
    #    model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return d[str(ind.item())], sm_output[ind] #([predicted class, description], probability)

def benchmark(model, device, backend, input_shape=(1024, 1, 224, 224), dtype='FP32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    if backend == 'ov':
        provider_options = OpenVINOProviderOptions(backend = device, precision = dtype) 
        model = ORTInferenceModule(model, provider_options = provider_options)
    else:
        input_data = input_data.to("cuda")
        if dtype=='fp16':
            input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)

    if backend == 'cuda':
        torch.cuda.synchronize()

    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            if args.backend == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

   
    print('\nAverage FPS: %.2f '%(1/np.mean(timings)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model path')
    parser.add_argument('--device', default='GPU', help='device type : GPU | CPU')
    parser.add_argument('--backend', default='ov', help='backend ov | cuda')
    parser.add_argument('--dtype', default='FP32', help='data type : FP32 | FP16')

    args = parser.parse_args()

    model = models.resnet50(pretrained=True)
    print("Loading model ", args.model)
    model.load_state_dict(torch.load(args.model))
    
    benchmark(model, args.device, args.backend, input_shape=(1, 3, 224, 224), dtype=args.dtype, nruns=100)




