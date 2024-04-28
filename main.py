
import torch
from model import *
from torch.nn.parallel import DataParallel
from utills import *
pre_trained=True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device=device)

model.to(device)
eval_data_dir = 'datasets/Dataset_Speech_Assignment'
fine_tune_data_dir = 'datasets/for-2seconds/training'

if pre_trained:
    file_path = "models/Best_LA_model_for_DF.pth"
    dictModel = torch.load(file_path,map_location=device)
    model = DataParallel(model)
    model.load_state_dict(dictModel)
    model_name = "pre-trained"
else:
    print("Training the model")
    #check if already fine-tuned model exists
    fine_tunemodel_path = "models/fine_tuned.pth"
    if os.path.exists(fine_tunemodel_path):
        model.load_state_dict(torch.load(fine_tunemodel_path))
        model = model.to(device)
        
    else:
        traindataset = FORDataset("datasets/for-2seconds/training")
        train_loader = DataLoader(traindataset, batch_size=14, shuffle=True)
        evaldataset = FORDataset("datasets/for-2seconds/validation")
        eval_loader = DataLoader(evaldataset, batch_size=14, shuffle=True)
        model = finetune(train_loader,eval_loader, model, device,lr=0.001, epochs=5)
        # save fine-tuned model
        torch.save(model.state_dict(), fine_tunemodel_path)
    model_name = "fine-tuned"


# Load the evaluation data
customdataset = CustomDataset(eval_data_dir)
customdataloader = DataLoader(customdataset, batch_size=32, shuffle=True)

# Evaluate the model
all_embeddings, all_labels, accuracy, genuine, impostor = eval(customdataloader, model, device)
accuracy_percentage = "{:.4%}".format(accuracy)
eer, eer_threshold =calculate_eer(genuine, impostor)
print(f"EER: {eer}")
print(f"EER Threshold: {eer_threshold}")
auc = calculate_auc(all_embeddings, all_labels,accuracy_percentage,round(eer,3),round(eer_threshold,3),model_name=model_name)
print(f"Accuracy: {accuracy_percentage}")
print(f"AUC: {auc}")