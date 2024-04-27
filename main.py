
import torch
from model import *
from torch.nn.parallel import DataParallel
from utills import *
pre_trained=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device=device)
model = DataParallel(model)
model.to(device)
if pre_trained:
    file_path = "models/Best_LA_model_for_DF.pth"
    dictModel = torch.load(file_path,map_location=device)
    model.load_state_dict(dictModel)
else:
    print("Training the model")
    #model = fine_tune(model, device)

data_dir = 'datasets/Dataset_Speech_Assignment'
customdataset = CustomDataset(data_dir)
customdataloader = DataLoader(customdataset, batch_size=32, shuffle=True)
# evaluation_results = evaluate(model, customdataloader, device)
