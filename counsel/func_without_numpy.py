import csv

from apps.shop.models import Picture


def similar_csv_to_db(path_to_csv, picture_id_to_matrix):
    with open(path_to_csv, mode='r') as f:
        reader = csv.reader(f)
        data = list(reader)

    reverse_picture_id_to_matrix = {v: int(k) for k, v in picture_id_to_matrix.items()}
    for idx, val in enumerate(data):
        ids = list(map(int, val))
        pic = Picture.objects.get(id=reverse_picture_id_to_matrix[idx])
        pic.similar_pictures.add(*ids)
