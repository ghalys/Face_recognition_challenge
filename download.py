import os
import requests
from bs4 import BeautifulSoup

def download_images(query, num_images, folder_path):
    print("okk")
    query = query.replace(' ', '+')  # Remplacer les espaces par '+'
    url = f"https://www.google.com/search?hl=en&q={query}&tbm=isch"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    images = [img['src'] for img in soup.find_all('img')]

    # Définir le nom du dossier où sauvegarder les images
    folder_name = os.path.join(folder_path, query.replace('+', '_'))
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    # Télécharger les premières 'num_images' images
    for i, img_url in enumerate(images[:num_images]):
        try:
            img_data = requests.get(img_url).content
            with open(os.path.join(folder_name, f'image_{i+1}.jpg'), 'wb') as file:
                file.write(img_data)
            print(f"Image {i+1} téléchargée dans {folder_name}")
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image {i+1}: {e}")

# Exemple d'utilisation
download_images("kellan lutz", 5, "Dataset")
