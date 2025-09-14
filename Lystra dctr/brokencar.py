import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import json

# --- 1. Загружаем модель ---
NUM_CLASSES = 5  # замени на своё количество классов
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("fasterrcnn_resnet50_multidataset.pth", map_location="cpu"))
model.eval()

# --- 2. Трансформация для картинки ---
transform = transforms.Compose([
    transforms.ToTensor()
])

# --- 3. Загружаем тестовое изображение ---
image_path = input("напишите название файла, например test.jpg: ")   # сюда подставь свой файл
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

# --- 4. Прогон через модель ---
with torch.no_grad():
    outputs = model(img_tensor)

threshold = 0.5  # показываем только объекты с score > 0.5

# --- 5. Маппинг индекса на название дефекта ---
INDEX_TO_CLASS = {
    1: "car",
    2: "dunt",
    3: "rust",
    4: "scratch"
}

filtered_boxes = []
filtered_labels = []
filtered_scores = []

for box, label, score in zip(outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]):
    if score >= threshold:
        filtered_boxes.append(box.cpu().numpy().tolist())
        filtered_labels.append(INDEX_TO_CLASS.get(label.item(), "unknown"))
        filtered_scores.append(score.item())

# --- 6. Формируем JSON ---
result = {
    "boxes": filtered_boxes,
    "labels": filtered_labels,
    "scores": filtered_scores
}

print(json.dumps(result, indent=2, ensure_ascii=False))