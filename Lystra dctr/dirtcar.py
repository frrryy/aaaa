import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 🔹 Классы (в том же порядке, как в обучении!)
class_names = ["clean", "dirty", "slightly dirty", "super clean", "super dirty"]

# 🔹 Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🔹 Загружаем модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("resnet50_dirtycar.pth", map_location=device))
model = model.to(device)
model.eval()

# 🔹 Функция предсказания
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)  # вероятности
        conf, pred = torch.max(probs, 1)

    # возвращаем предсказанный класс и все проценты
    result = {
        "predicted": class_names[pred.item()],
        "confidence": round(conf.item() * 100, 2),
        "all_classes": {class_names[i]: round(probs[0][i].item() * 100, 2) for i in range(len(class_names))}
    }
    return result

# 🔹 Тест
res = predict(input("напишите название файла, например test.jpg: "))
print("Предсказание:", res["predicted"])
print("Уверенность:", res["confidence"], "%")
print("Все классы:", res["all_classes"])
