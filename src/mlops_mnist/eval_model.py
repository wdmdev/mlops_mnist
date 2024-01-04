import os

import torch
from tqdm import tqdm

def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    model = torch.load(model_checkpoint)
    test_set = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test_loader.pt"))

    accuracy = 0
    with torch.no_grad():
        for images, labels in tqdm(test_set, total=len(test_set)):
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Accuracy: {accuracy/len(test_set)}")