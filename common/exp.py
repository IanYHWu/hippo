import torch
import torch.nn.functional as F
import numpy as np






if __name__ == "__main__":
    a = torch.tensor([[[[1, 2, 3], [5, 5, 5]], [[4, 5, 6], [9, 9, 9]]],
                      [[[0, 2, 3], [5, 8, 5]], [[4, 5, 6], [9, 10, 9]]]])

    b = np.array([np.random.randint(3)])
    print(b)


