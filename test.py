if __name__ == "__main__":
    from pytorch_retinanet.models import Retinanet
    import torch

    model = Retinanet(num_classes=5)
    model.eval()
    rnd = torch.randn([1, 3, 600, 600])
    outputs = model(rnd)
    print(outputs)
