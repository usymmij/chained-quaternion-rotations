import torch

# set default to cuda if available

# :( turns out the dataset is generaated faster on cpu lol

#device = torch.device("cuda:0" if torch.cuda.is_available() 
#                      else "cpu")
#torch.set_default_device(device)
